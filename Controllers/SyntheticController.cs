using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks.SyntheticData;
using AiDotNet.Tensors.LinearAlgebra;
using Microsoft.AspNetCore.Mvc;
using System.Diagnostics;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Text.Json.Serialization;

namespace AiDotNetPocApi.Controllers;

// Synthetic data is information that's artificially created rather than collected from the real world.
// Instead of gathering actual emails from real inboxes, you generate fake-but-realistic examples yourself — like the five spam messages
// which were made up to look and behave like the real thing.
// People use synthetic data when real data is scarce, private, expensive to collect, or doesn't cover enough tricky cases.
// It lets you deliberately create the exact examples you need,
// such as rare or hard-to-find situations, to test or train a model more thoroughly.
// The main catch is that synthetic data is only as good as the process that made it:
// if it doesn't capture the messiness and variety of real data,
// a model trained on it may struggle once it faces the real world.


// ─── Facade interface ─────────────────────────────────────────────────────────
// Hides all AiDotNet types from the controller; controller depends only on this.

public interface ISyntheticDataFacade
{
    Task<SyntheticGenerationResult> GenerateAsync(int numSamples, CancellationToken ct = default);
}

// ─── Domain result ────────────────────────────────────────────────────────────

public sealed record SyntheticGenerationResult(
    List<List<double>> Rows,
    List<ColumnStats> RealStats,
    List<ColumnStats> SyntheticStats,
    List<CorrelationStats> Correlations,
    double FitMs,
    double GenerateMs,
    string FacadePattern,
    string InterfaceChain);

// ─── Facade implementation ────────────────────────────────────────────────────
// Owns the AiDotNet generator lifecycle. Registered as a SINGLETON: the generator
// is fitted exactly once (lazily, on first request, off the request thread) and
// reused for all subsequent generation.
//
// Model: SMOTENCGenerator<double> via the synthetic-data module's first-class API:
//
//   ISyntheticTabularGenerator<T>.Fit(Matrix, IReadOnlyList<ColumnMetadata>, epochs)
//   → Generate(numSamples)
//
// SMOTE-NC interpolates between nearest-neighbor seed rows, handling the
// categorical Education column natively (the schema is actually passed to Fit —
// "NC" = Nominal + Continuous).
//
// Why SMOTENC and not CTGANGenerator (all verified empirically on 0.213.3 with
// this 50-row seed set):
// - CTGAN's own Fit/Generate runs, but the GAN does not converge on small data:
//   at 100 epochs (17s) generated ranges overshoot badly (negative incomes, ages
//   to 96, credit-score std 3× real); at 300 epochs (63s) it diverges outright
//   (income std 188k vs real 32k, credit scores −2214..2142). More epochs = worse.
// - CopulaSynthGenerator fits instantly with good marginals, but pairwise
//   correlations degrade (0.82–0.93 vs real 0.95–0.99) and categorical values
//   come back with float noise (Education = 1.0000003).
// - SMOTENCGenerator fits instantly and preserves both marginals AND pairwise
//   correlations (synthetic 0.989/0.956/0.972 vs real 0.989/0.954/0.968), with
//   exact-integer categorical values. For a 50-row seed set, neighborhood
//   interpolation is simply the right tool.
// - The previous AiModelBuilder.BuildAsync + Predict(noise) path never invoked
//   CTGAN's adversarial Fit at all — the schema was built but never passed, and
//   generation went through the generic NeuralNetworkBase.Predict contract.

public sealed class SmoteSyntheticDataFacade : ISyntheticDataFacade
{
    public const int SeedRows = 50;
    public const int FeatureCount = 4;
    public const int KNeighbors = 5;
    private const int Seed = 42;

    public static readonly string[] ColumnNames = ["Age", "Income", "Education", "CreditScore"];

    // Customer profile schema: Age, Income, Education (categorical), CreditScore.
    // Passed to Fit so the generator treats Education as nominal, not continuous.
    public static readonly ColumnMetadata[] Schema =
    [
        new ColumnMetadata("Age",         ColumnDataType.Continuous),
        new ColumnMetadata("Income",      ColumnDataType.Continuous),
        new ColumnMetadata("Education",   ColumnDataType.Categorical,
            ["HighSchool", "Bachelor", "Master", "PhD"]),
        new ColumnMetadata("CreditScore", ColumnDataType.Continuous),
    ];

    private static readonly Matrix<double> SeedData = BuildSeedData();
    private static readonly List<ColumnStats> RealStats = ComputeStats(SeedData);

    // Fitted once per process; all requests share the result.
    private readonly Lazy<Task<FittedGenerator>> _generator = new(
        () => Task.Run(FitGenerator),
        LazyThreadSafetyMode.ExecutionAndPublication);

    // Generate advances the generator's internal RNG; serialize access.
    private readonly SemaphoreSlim _generateLock = new(1, 1);

    private sealed record FittedGenerator(SMOTENCGenerator<double> Generator, double FitMs);

    public async Task<SyntheticGenerationResult> GenerateAsync(
        int numSamples, CancellationToken ct = default)
    {
        // Fitting is shared state — never cancelled by an individual request;
        // the request just stops waiting for it.
        var fitted = await _generator.Value.WaitAsync(ct);

        var generateTimer = Stopwatch.StartNew();

        Matrix<double> synthetic;
        await _generateLock.WaitAsync(ct);
        try
        {
            synthetic = fitted.Generator.Generate(numSamples);
        }
        finally
        {
            _generateLock.Release();
        }

        generateTimer.Stop();

        return new SyntheticGenerationResult(
            Rows: MatrixToRows(synthetic),
            RealStats: RealStats,
            SyntheticStats: ComputeStats(synthetic),
            Correlations: ComputeCorrelations(SeedData, synthetic),
            FitMs: fitted.FitMs,
            GenerateMs: generateTimer.Elapsed.TotalMilliseconds,
            FacadePattern:
                "ISyntheticDataFacade → SmoteSyntheticDataFacade (singleton, fitted once)" +
                $" | new SMOTENCGenerator<double>(SMOTENCOptions(K={KNeighbors}, Seed={Seed}))" +
                $" → Fit(Matrix[{SeedRows},{FeatureCount}], ColumnMetadata[{FeatureCount}] (Education categorical), epochs:1)" +
                " → Generate(numSamples) → Matrix[numSamples,4] (categorical column decoded to exact category indices)",
            InterfaceChain:
                "SMOTENCGenerator<T> → SyntheticTabularGeneratorBase<T> → ISyntheticTabularGenerator<T>");
    }

    // ─── One-time fit ─────────────────────────────────────────────────────────

    private static FittedGenerator FitGenerator()
    {
        var timer = Stopwatch.StartNew();

        var generator = new SMOTENCGenerator<double>(new SMOTENCOptions<double>
        {
            K = KNeighbors,
            Seed = Seed,
        });
        generator.Fit(SeedData, Schema, epochs: 1); // epochs is unused by SMOTE-NC

        timer.Stop();
        return new FittedGenerator(generator, timer.Elapsed.TotalMilliseconds);
    }

    // ─── Seed data ────────────────────────────────────────────────────────────

    private static Matrix<double> BuildSeedData()
    {
        // 50 customer profiles: Age, Income, Education (0-3 ordinal), CreditScore
        // Education encoding: 0=HighSchool, 1=Bachelor, 2=Master, 3=PhD
        double[,] rows =
        {
            { 28, 42000, 1, 640 }, { 35, 68000, 2, 710 }, { 52, 95000, 3, 780 },
            { 24, 31000, 0, 590 }, { 41, 85000, 2, 750 }, { 33, 56000, 1, 680 },
            { 47, 110000, 3, 820 }, { 29, 38000, 1, 620 }, { 38, 72000, 2, 740 },
            { 55, 130000, 3, 800 }, { 22, 28000, 0, 560 }, { 44, 92000, 2, 770 },
            { 31, 48000, 1, 650 }, { 60, 125000, 3, 810 }, { 26, 35000, 0, 610 },
            { 39, 79000, 2, 730 }, { 50, 105000, 3, 790 }, { 27, 41000, 1, 630 },
            { 34, 63000, 1, 700 }, { 43, 88000, 2, 760 }, { 25, 33000, 0, 580 },
            { 48, 98000, 3, 785 }, { 36, 70000, 2, 720 }, { 53, 115000, 3, 795 },
            { 30, 45000, 1, 645 }, { 42, 87000, 2, 755 }, { 23, 29000, 0, 570 },
            { 57, 120000, 3, 805 }, { 32, 52000, 1, 670 }, { 45, 93000, 2, 765 },
            { 21, 26000, 0, 550 }, { 37, 74000, 2, 735 }, { 49, 102000, 3, 788 },
            { 28, 40000, 1, 638 }, { 40, 82000, 2, 748 }, { 54, 118000, 3, 798 },
            { 26, 36000, 0, 605 }, { 35, 67000, 2, 715 }, { 46, 96000, 3, 782 },
            { 31, 47000, 1, 652 }, { 59, 128000, 3, 812 }, { 24, 32000, 0, 575 },
            { 38, 76000, 2, 738 }, { 51, 108000, 3, 792 }, { 27, 39000, 1, 625 },
            { 43, 90000, 2, 762 }, { 22, 27000, 0, 555 }, { 56, 122000, 3, 802 },
            { 33, 58000, 1, 688 }, { 41, 84000, 2, 752 },
        };

        return new Matrix<double>(rows);
    }

    // ─── Row / statistics helpers ─────────────────────────────────────────────

    private static List<List<double>> MatrixToRows(Matrix<double> matrix)
    {
        var rows = new List<List<double>>(matrix.Rows);
        for (int r = 0; r < matrix.Rows; r++)
        {
            var row = new List<double>(matrix.Columns);
            for (int c = 0; c < matrix.Columns; c++)
                row.Add(Math.Round(matrix[r, c], 4));
            rows.Add(row);
        }
        return rows;
    }

    private static List<ColumnStats> ComputeStats(Matrix<double> data)
    {
        var stats = new List<ColumnStats>(FeatureCount);
        for (int col = 0; col < Math.Min(FeatureCount, data.Columns); col++)
        {
            var values = Enumerable.Range(0, data.Rows).Select(r => data[r, col]).ToArray();
            double mean = values.Average();
            double variance = values.Select(v => (v - mean) * (v - mean)).Average();

            stats.Add(new ColumnStats(
                Column: Schema[col].Name,
                Min: Math.Round(values.Min(), 3),
                Max: Math.Round(values.Max(), 3),
                Mean: Math.Round(mean, 3),
                Std: Math.Round(Math.Sqrt(variance), 3)));
        }
        return stats;
    }

    // Pearson correlation for every column pair, real vs synthetic — the headline
    // fidelity metric for synthetic tabular data.
    private static List<CorrelationStats> ComputeCorrelations(
        Matrix<double> real, Matrix<double> synthetic)
    {
        var result = new List<CorrelationStats>();
        for (int a = 0; a < FeatureCount; a++)
            for (int b = a + 1; b < FeatureCount; b++)
                result.Add(new CorrelationStats(
                    Pair: $"{ColumnNames[a]}~{ColumnNames[b]}",
                    Real: Math.Round(PearsonCorrelation(real, a, b), 4),
                    Synthetic: Math.Round(PearsonCorrelation(synthetic, a, b), 4)));
        return result;
    }

    private static double PearsonCorrelation(Matrix<double> m, int a, int b)
    {
        int n = m.Rows;
        double meanA = 0, meanB = 0;
        for (int r = 0; r < n; r++) { meanA += m[r, a]; meanB += m[r, b]; }
        meanA /= n; meanB /= n;

        double cov = 0, varA = 0, varB = 0;
        for (int r = 0; r < n; r++)
        {
            cov += (m[r, a] - meanA) * (m[r, b] - meanB);
            varA += Math.Pow(m[r, a] - meanA, 2);
            varB += Math.Pow(m[r, b] - meanB, 2);
        }
        double denom = Math.Sqrt(varA * varB);
        return denom < 1e-12 ? 0 : cov / denom;
    }
}

// ─── Controller ───────────────────────────────────────────────────────────────
// Thin HTTP layer: validates input, delegates to ISyntheticDataFacade, shapes response.

[ApiController]
[Route("api/[controller]")]
public sealed class SyntheticController : ControllerBase
{
    private readonly ISyntheticDataFacade _facade;

    public SyntheticController(ISyntheticDataFacade facade) => _facade = facade;

    /// <summary>
    /// POC: Synthetic tabular data generation using SMOTENCGenerator behind
    /// ISyntheticDataFacade. The generator is fitted once per process via the
    /// synthetic-data module's first-class ISyntheticTabularGenerator API
    /// (Fit with explicit ColumnMetadata so Education is treated as categorical)
    /// and cached; requests pay generation cost only (~ms). The response includes
    /// per-column stats and pairwise correlations, real vs synthetic.
    /// </summary>
    [HttpPost("Generate")]
    public async Task<ActionResult<SyntheticPocResponse>> Generate(
        [FromBody] SyntheticRequest request,
        CancellationToken cancellationToken)
    {
        int numSamples = Math.Clamp(request.NumSamples ?? 20, 1, 200);

        var totalTimer = Stopwatch.StartNew();
        var result = await _facade.GenerateAsync(numSamples, cancellationToken);
        totalTimer.Stop();

        return Ok(new SyntheticPocResponse(
            SyntheticSamples: result.Rows,
            RealStats: result.RealStats,
            SyntheticStats: result.SyntheticStats,
            Correlations: result.Correlations,
            ModelInfo: new SyntheticModelInfo(
                ModelType: "SMOTENCGenerator",
                KNeighbors: SmoteSyntheticDataFacade.KNeighbors,
                SeedRows: SmoteSyntheticDataFacade.SeedRows,
                GeneratedSamples: numSamples,
                ColumnSchema: SmoteSyntheticDataFacade.Schema.Select(c => new ColumnSchema(
                    c.Name, c.DataType.ToString(), c.Categories?.ToList())).ToList()),
            FacadePattern: result.FacadePattern,
            InterfaceChain: result.InterfaceChain,
            Timings: new SyntheticTimings(
                TotalMs: Math.Round(totalTimer.Elapsed.TotalMilliseconds, 3),
                TrainMs: Math.Round(result.FitMs, 3),
                InferenceMs: Math.Round(result.GenerateMs, 3)),
            System: BuildSystemInfo()));
    }

    private static PocSystemInfo BuildSystemInfo() => new(
        Os: RuntimeInformation.OSDescription,
        Framework: RuntimeInformation.FrameworkDescription,
        // NB: AiDotNet 0.213.3 ships with stale assembly version attributes (0.204.0);
        // this reports what the loaded assembly declares about itself.
        LibraryVersion: typeof(SMOTENCGenerator<>).Assembly
            .GetCustomAttribute<AssemblyInformationalVersionAttribute>()?.InformationalVersion
            ?? "unknown");
}

// ─── Request / Response contracts ────────────────────────────────────────────

public sealed record SyntheticRequest(
    [property: JsonPropertyName("numSamples")] int? NumSamples);

public sealed record SyntheticPocResponse(
    [property: JsonPropertyName("syntheticSamples")] List<List<double>> SyntheticSamples,
    [property: JsonPropertyName("realStats")] List<ColumnStats> RealStats,
    [property: JsonPropertyName("syntheticStats")] List<ColumnStats> SyntheticStats,
    [property: JsonPropertyName("correlations")] List<CorrelationStats> Correlations,
    [property: JsonPropertyName("modelInfo")] SyntheticModelInfo ModelInfo,
    [property: JsonPropertyName("facadePattern")] string FacadePattern,
    [property: JsonPropertyName("interfaceChain")] string InterfaceChain,
    [property: JsonPropertyName("timings")] SyntheticTimings Timings,
    [property: JsonPropertyName("system")] PocSystemInfo System);

public sealed record ColumnStats(
    [property: JsonPropertyName("column")] string Column,
    [property: JsonPropertyName("min")] double Min,
    [property: JsonPropertyName("max")] double Max,
    [property: JsonPropertyName("mean")] double Mean,
    [property: JsonPropertyName("std")] double Std);

public sealed record CorrelationStats(
    [property: JsonPropertyName("pair")] string Pair,
    [property: JsonPropertyName("real")] double Real,
    [property: JsonPropertyName("synthetic")] double Synthetic);

public sealed record SyntheticModelInfo(
    [property: JsonPropertyName("modelType")] string ModelType,
    [property: JsonPropertyName("kNeighbors")] int KNeighbors,
    [property: JsonPropertyName("seedRows")] int SeedRows,
    [property: JsonPropertyName("generatedSamples")] int GeneratedSamples,
    [property: JsonPropertyName("columnSchema")] List<ColumnSchema> ColumnSchema);

public sealed record ColumnSchema(
    [property: JsonPropertyName("name")] string Name,
    [property: JsonPropertyName("dataType")] string DataType,
    [property: JsonPropertyName("categories")] List<string>? Categories);

public sealed record SyntheticTimings(
    [property: JsonPropertyName("totalMs")] double TotalMs,
    [property: JsonPropertyName("trainMs")] double TrainMs,
    [property: JsonPropertyName("inferenceMs")] double InferenceMs);
