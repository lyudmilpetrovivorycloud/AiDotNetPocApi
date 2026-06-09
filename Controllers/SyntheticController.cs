using AiDotNet;
using AiDotNet.Data.Loaders;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.SyntheticData;
using AiDotNet.Tensors.LinearAlgebra;
using Microsoft.AspNetCore.Mvc;
using System.Diagnostics;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Text.Json.Serialization;

namespace AiDotNetPocApi.Controllers;

[ApiController]
[Route("api/[controller]")]
public sealed class SyntheticController : ControllerBase
{
    // Customer profile schema: Age, Income, Education (categorical), CreditScore
    private static readonly ColumnMetadata[] Schema =
    [
        new ColumnMetadata("Age",         ColumnDataType.Continuous),
        new ColumnMetadata("Income",      ColumnDataType.Continuous),
        new ColumnMetadata("Education",   ColumnDataType.Categorical,
            ["HighSchool", "Bachelor", "Master", "PhD"]),
        new ColumnMetadata("CreditScore", ColumnDataType.Continuous),
    ];

    private const int FeatureCount = 4; // matches Schema length
    private const int SeedRows = 50;

    // 50 realistic customer profiles (seed=7 for reproducibility)
    private static readonly Matrix<double> SeedData = BuildSeedData();

    /// <summary>
    /// POC: Synthetic tabular data generation using CTGANGenerator via AiDotNet facade.
    /// CTGANGenerator extends NeuralNetworkBase&lt;T&gt; which implements
    /// IFullModel&lt;T, Tensor&lt;T&gt;, Tensor&lt;T&gt;&gt;, so it integrates with
    /// AiModelBuilder&lt;double, Tensor&lt;double&gt;, Tensor&lt;double&gt;&gt; directly.
    /// Demonstrates DataLoaders.FromTensors → AiModelBuilder.ConfigureModel → BuildAsync → Predict.
    /// </summary>
    [HttpPost("Generate")]
    public async Task<ActionResult<SyntheticPocResponse>> Generate(
        [FromBody] SyntheticRequest request,
        CancellationToken cancellationToken)
    {
        int numSamples = Math.Clamp(request.NumSamples ?? 20, 1, 200);

        var totalTimer = Stopwatch.StartNew();
        var trainTimer = Stopwatch.StartNew();

        // ─────────────────────────────────────────────────────────────────
        // 1. Convert seed Matrix to Tensor and wrap in data loader
        //    Shape: [SeedRows, FeatureCount] — unsupervised, input == target
        // ─────────────────────────────────────────────────────────────────
        var seedTensor = MatrixToTensor(SeedData);
        var dataLoader = DataLoaders.FromTensors(seedTensor, seedTensor);

        // ─────────────────────────────────────────────────────────────────
        // 2. Configure CTGANGenerator
        //    CTGANGenerator → NeuralNetworkBase<T> → IFullModel<T, Tensor<T>, Tensor<T>>
        // ─────────────────────────────────────────────────────────────────
        var architecture = new NeuralNetworkArchitecture<double>(
            inputFeatures: FeatureCount,
            outputSize: FeatureCount);

        var ctganOptions = new CTGANOptions<double>
        {
            EmbeddingDimension = 64,
            GeneratorDimensions = [128, 128],
            DiscriminatorDimensions = [128, 128],
            BatchSize = Math.Min(SeedRows, 16),
        };

        var generator = new CTGANGenerator<double>(architecture, ctganOptions);

        // Full facade path: ConfigureDataLoader + ConfigureModel + BuildAsync
        var result = await new AiModelBuilder<double, Tensor<double>, Tensor<double>>()
            .ConfigureDataLoader(dataLoader)
            .ConfigureModel(generator)
            .BuildAsync();

        trainTimer.Stop();

        // ─────────────────────────────────────────────────────────────────
        // 3. Generate synthetic rows via AiModelResult.Predict
        //    Noise input shape: [numSamples, EmbeddingDim] — Gaussian noise seed
        // ─────────────────────────────────────────────────────────────────
        var inferenceTimer = Stopwatch.StartNew();

        var noiseTensor = BuildNoiseTensor(numSamples, embeddingDim: 64);
        var syntheticTensor = result.Predict(noiseTensor);

        inferenceTimer.Stop();
        totalTimer.Stop();

        // ─────────────────────────────────────────────────────────────────
        // 4. Collect comparison statistics (real vs synthetic)
        // ─────────────────────────────────────────────────────────────────
        var realStats = ComputeStats(SeedData);
        var syntheticStats = ComputeTensorStats(syntheticTensor, numSamples);
        var syntheticRows = TensorToRows(syntheticTensor, numSamples);

        return Ok(new SyntheticPocResponse(
            SyntheticSamples: syntheticRows,
            RealStats: realStats,
            SyntheticStats: syntheticStats,
            ModelInfo: new SyntheticModelInfo(
                ModelType: "CTGANGenerator",
                EmbeddingDimension: 64,
                GeneratorDimensions: [128, 128],
                DiscriminatorDimensions: [128, 128],
                SeedRows: SeedRows,
                GeneratedSamples: numSamples,
                ColumnSchema: Schema.Select(c => new ColumnSchema(
                    c.Name, c.DataType.ToString(), c.Categories?.ToList())).ToList()),
            FacadePattern: "AiModelBuilder<double, Tensor<double>, Tensor<double>>" +
                           ".ConfigureDataLoader(DataLoaders.FromTensors(...))" +
                           ".ConfigureModel(CTGANGenerator<double>)" +
                           ".BuildAsync() → AiModelResult.Predict(Tensor<double>)",
            InterfaceChain: "CTGANGenerator → NeuralNetworkBase<T> → IFullModel<T, Tensor<T>, Tensor<T>>",
            Timings: new SyntheticTimings(
                TotalMs: Math.Round(totalTimer.Elapsed.TotalMilliseconds, 3),
                TrainMs: Math.Round(trainTimer.Elapsed.TotalMilliseconds, 3),
                InferenceMs: Math.Round(inferenceTimer.Elapsed.TotalMilliseconds, 3)),
            System: BuildSystemInfo()));
    }

    // ─── Seed data ────────────────────────────────────────────────────────

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

    // ─── Tensor helpers ───────────────────────────────────────────────────

    private static Tensor<double> MatrixToTensor(Matrix<double> matrix)
    {
        int rows = matrix.Rows, cols = matrix.Columns;
        var values = new double[rows * cols];
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                values[r * cols + c] = matrix[r, c];
        return new Tensor<double>(values, [rows, cols]);
    }

    // Gaussian noise seed for the generator (Box-Muller transform)
    private static Tensor<double> BuildNoiseTensor(int numSamples, int embeddingDim)
    {
        var rng = new Random(42);
        var values = new double[numSamples * embeddingDim];
        for (int i = 0; i < values.Length; i += 2)
        {
            double u1 = 1.0 - rng.NextDouble();
            double u2 = 1.0 - rng.NextDouble();
            double z = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
            values[i] = z;
            if (i + 1 < values.Length)
                values[i + 1] = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
        }
        return new Tensor<double>(values, [numSamples, embeddingDim]);
    }

    private static List<List<double>> TensorToRows(Tensor<double> tensor, int limit)
    {
        // Expect shape [numSamples, FeatureCount] or [numSamples, FeatureCount, ...]
        int actualRows = tensor.Shape[0];
        int cols = tensor.Shape.Length > 1 ? tensor.Shape[1] : 1;
        int n = Math.Min(limit, actualRows);

        var rows = new List<List<double>>();
        for (int r = 0; r < n; r++)
        {
            var row = new List<double>();
            for (int c = 0; c < cols; c++)
                row.Add(Math.Round(tensor[r, c], 4));
            rows.Add(row);
        }
        return rows;
    }

    // ─── Statistics ───────────────────────────────────────────────────────

    private static List<ColumnStats> ComputeStats(Matrix<double> data)
    {
        int rows = data.Rows;
        var stats = new List<ColumnStats>();

        for (int col = 0; col < Math.Min(FeatureCount, data.Columns); col++)
        {
            var values = Enumerable.Range(0, rows).Select(r => data[r, col]).ToArray();
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

    private static List<ColumnStats> ComputeTensorStats(Tensor<double> tensor, int numRows)
    {
        int rows = Math.Min(numRows, tensor.Shape[0]);
        int cols = Math.Min(FeatureCount, tensor.Shape.Length > 1 ? tensor.Shape[1] : 1);
        var stats = new List<ColumnStats>();

        for (int col = 0; col < cols; col++)
        {
            var values = Enumerable.Range(0, rows).Select(r => tensor[r, col]).ToArray();
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

    // ─── System info ─────────────────────────────────────────────────────

    private static PocSystemInfo BuildSystemInfo() => new(
        Os: RuntimeInformation.OSDescription,
        Framework: RuntimeInformation.FrameworkDescription,
        LibraryVersion: typeof(AiModelBuilder<,,>).Assembly
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

public sealed record SyntheticModelInfo(
    [property: JsonPropertyName("modelType")] string ModelType,
    [property: JsonPropertyName("embeddingDimension")] int EmbeddingDimension,
    [property: JsonPropertyName("generatorDimensions")] int[] GeneratorDimensions,
    [property: JsonPropertyName("discriminatorDimensions")] int[] DiscriminatorDimensions,
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
