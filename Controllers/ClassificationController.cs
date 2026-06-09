using AiDotNet;
using AiDotNet.Classification.Ensemble;
using AiDotNet.Data.Loaders;
using AiDotNet.Models.Options;
using AiDotNet.Tensors.LinearAlgebra;
using Microsoft.AspNetCore.Mvc;
using System.Diagnostics;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Text.Json.Serialization;

namespace AiDotNetPocApi.Controllers;

[ApiController]
[Route("api/[controller]")]
public sealed class ClassificationController : ControllerBase
{
    // Spam detection feature schema (8 features)
    // [0] wordCount [1] capsRatio [2] exclamations [3] urlCount
    // [4] moneyKeywords [5] urgencyKeywords [6] linkRatio [7] avgWordLen
    private const int FeatureCount = 8;
    private const int TrainSamples = 100;

    // 100 deterministic training samples (seed=42): 50 spam (label=1), 50 not-spam (label=0)
    // Generated with a seeded RNG so results are reproducible across runs.
    private static readonly (double[,] Features, double[] Labels) TrainingData = GenerateTrainingData();

    /// <summary>
    /// POC: Binary spam classification using RandomForestClassifier via the full AiDotNet facade.
    /// Demonstrates DataLoaders.FromArrays → AiModelBuilder.ConfigureModel → BuildAsync → Predict.
    /// </summary>
    [HttpPost("Predict")]
    public async Task<ActionResult<ClassificationPocResponse>> Predict(
        [FromBody] ClassificationRequest request,
        CancellationToken cancellationToken)
    {
        if (request.Samples is null || request.Samples.Count == 0)
            return BadRequest(new { error = "Provide at least one sample to classify." });

        var totalTimer = Stopwatch.StartNew();
        var trainTimer = Stopwatch.StartNew();

        // ─────────────────────────────────────────────────────────────────
        // 1. Wrap training data in AiDotNet's InMemoryDataLoader
        // ─────────────────────────────────────────────────────────────────
        var dataLoader = DataLoaders.FromArrays(TrainingData.Features, TrainingData.Labels);

        // ─────────────────────────────────────────────────────────────────
        // 2. Configure RandomForestClassifier via AiModelBuilder facade
        //    RandomForestClassifier<T> implements IClassifier<T>
        //    which is IFullModel<T, Matrix<T>, Vector<T>> — the exact
        //    type expected by AiModelBuilder<double, Matrix<double>, Vector<double>>.
        // ─────────────────────────────────────────────────────────────────

        // 
        var classifier = new RandomForestClassifier<double>(
            new RandomForestClassifierOptions<double> { NEstimators = 50 });

        // Full facade path: ConfigureDataLoader + ConfigureModel + BuildAsync
        var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
            .ConfigureDataLoader(dataLoader)
            .ConfigureModel(classifier)
            .BuildAsync();

        trainTimer.Stop();

        // ─────────────────────────────────────────────────────────────────
        // 3. Predict on each incoming sample (AiModelResult.Predict)
        // ─────────────────────────────────────────────────────────────────
        var inferenceTimer = Stopwatch.StartNew();
        var predictions = new List<SamplePrediction>();

        foreach (var sample in request.Samples)
        {
            if (sample.Features.Count != FeatureCount)
                return BadRequest(new
                {
                    error = $"Each sample must have exactly {FeatureCount} features: " +
                            "wordCount, capsRatio, exclamations, urlCount, " +
                            "moneyKeywords, urgencyKeywords, linkRatio, avgWordLen"
                });

            // AiModelResult<T, Matrix<T>, Vector<T>>.Predict(Matrix<T>)
            // Wrap the single sample as a 1-row matrix
            var featureMatrix = BuildSingleRowMatrix(sample.Features);
            var scores = result.Predict(featureMatrix);

            // scores is a Vector<double>; element 0 = class probability / score
            double rawScore = scores.Length > 0 ? scores[0] : 0.0;
            bool isSpam = rawScore >= 0.5;

            predictions.Add(new SamplePrediction(
                Features: sample.Features,
                Label: sample.Label,
                RawScore: Math.Round(rawScore, 6),
                Prediction: isSpam ? "spam" : "not-spam",
                Correct: sample.Label.HasValue
                    ? (sample.Label.Value == 1) == isSpam
                    : null));
        }

        inferenceTimer.Stop();
        totalTimer.Stop();

        double? accuracy = predictions.All(p => p.Correct.HasValue)
            ? Math.Round(predictions.Count(p => p.Correct == true) / (double)predictions.Count, 4)
            : null;

        return Ok(new ClassificationPocResponse(
            Predictions: predictions,
            ModelInfo: new ClassificationModelInfo(
                ModelType: "RandomForestClassifier",
                NumberOfTrees: 50,
                FeatureCount: FeatureCount,
                TrainSamples: TrainSamples,
                FeatureNames: ["wordCount", "capsRatio", "exclamations", "urlCount",
                                "moneyKeywords", "urgencyKeywords", "linkRatio", "avgWordLen"],
                ClassNames: ["not-spam", "spam"]),
            FacadePattern: "AiModelBuilder<double, Matrix<double>, Vector<double>>" +
                           ".ConfigureDataLoader(DataLoaders.FromArrays(...))" +
                           ".ConfigureModel(RandomForestClassifier<double>)" +
                           ".BuildAsync() → AiModelResult.Predict(Matrix<double>)",
            InterfaceChain: "RandomForestClassifier → IClassifier<T> → IFullModel<T, Matrix<T>, Vector<T>>",
            Accuracy: accuracy,
            Timings: new PocTimings(
                TotalMs: Math.Round(totalTimer.Elapsed.TotalMilliseconds, 3),
                TrainMs: Math.Round(trainTimer.Elapsed.TotalMilliseconds, 3),
                InferenceMs: Math.Round(inferenceTimer.Elapsed.TotalMilliseconds, 3)),
            System: BuildSystemInfo()));
    }

    // ─── Training data generation ─────────────────────────────────────────

    private static (double[,] Features, double[] Labels) GenerateTrainingData()
    {
        var rng = new Random(42);
        var features = new double[TrainSamples, FeatureCount];
        var labels = new double[TrainSamples];

        for (int i = 0; i < TrainSamples; i++)
        {
            bool isSpam = i < TrainSamples / 2;
            labels[i] = isSpam ? 1.0 : 0.0;

            if (isSpam)
            {
                // Spam: high caps, exclamations, money/urgency keywords, many URLs
                features[i, 0] = rng.Next(30, 80);                    // wordCount
                features[i, 1] = rng.NextDouble() * 0.4 + 0.3;        // capsRatio 0.3–0.7
                features[i, 2] = rng.Next(3, 10);                     // exclamations
                features[i, 3] = rng.Next(2, 6);                      // urlCount
                features[i, 4] = rng.Next(2, 5);                      // moneyKeywords
                features[i, 5] = rng.Next(2, 5);                      // urgencyKeywords
                features[i, 6] = rng.NextDouble() * 0.3 + 0.2;        // linkRatio 0.2–0.5
                features[i, 7] = rng.NextDouble() * 2.0 + 3.0;        // avgWordLen 3–5 (short)
            }
            else
            {
                // Legit: normal caps, few or no exclamations, no urgency patterns
                features[i, 0] = rng.Next(50, 300);                   // wordCount
                features[i, 1] = rng.NextDouble() * 0.1;              // capsRatio 0–0.1
                features[i, 2] = rng.Next(0, 2);                      // exclamations
                features[i, 3] = rng.Next(0, 2);                      // urlCount
                features[i, 4] = 0;                                   // moneyKeywords
                features[i, 5] = rng.Next(0, 1);                      // urgencyKeywords
                features[i, 6] = rng.NextDouble() * 0.05;             // linkRatio 0–0.05
                features[i, 7] = rng.NextDouble() * 2.5 + 5.0;        // avgWordLen 5–7.5 (longer)
            }
        }

        return (features, labels);
    }

    private static Matrix<double> BuildSingleRowMatrix(List<double> features)
    {
        var values = new double[1, features.Count];
        for (int j = 0; j < features.Count; j++)
            values[0, j] = features[j];
        return new Matrix<double>(values);
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

public sealed record ClassificationRequest(
    [property: JsonPropertyName("samples")] List<ClassificationSample> Samples);

public sealed record ClassificationSample(
    [property: JsonPropertyName("features")] List<double> Features,
    [property: JsonPropertyName("label")] int? Label);

public sealed record ClassificationPocResponse(
    [property: JsonPropertyName("predictions")] List<SamplePrediction> Predictions,
    [property: JsonPropertyName("modelInfo")] ClassificationModelInfo ModelInfo,
    [property: JsonPropertyName("facadePattern")] string FacadePattern,
    [property: JsonPropertyName("interfaceChain")] string InterfaceChain,
    [property: JsonPropertyName("accuracy")] double? Accuracy,
    [property: JsonPropertyName("timings")] PocTimings Timings,
    [property: JsonPropertyName("system")] PocSystemInfo System);

public sealed record SamplePrediction(
    [property: JsonPropertyName("features")] List<double> Features,
    [property: JsonPropertyName("label")] int? Label,
    [property: JsonPropertyName("rawScore")] double RawScore,
    [property: JsonPropertyName("prediction")] string Prediction,
    [property: JsonPropertyName("correct")] bool? Correct);

public sealed record ClassificationModelInfo(
    [property: JsonPropertyName("modelType")] string ModelType,
    [property: JsonPropertyName("numberOfTrees")] int NumberOfTrees,
    [property: JsonPropertyName("featureCount")] int FeatureCount,
    [property: JsonPropertyName("trainSamples")] int TrainSamples,
    [property: JsonPropertyName("featureNames")] string[] FeatureNames,
    [property: JsonPropertyName("classNames")] string[] ClassNames);
