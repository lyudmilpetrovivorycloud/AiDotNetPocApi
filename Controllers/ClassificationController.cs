using AiDotNet.NeuralNetworks.Tabular;
using AiDotNet.Models.Options;
using AiDotNet.Tensors.LinearAlgebra;
using Microsoft.AspNetCore.Mvc;
using System.Diagnostics;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Text.Json.Serialization;

namespace AiDotNetPocApi.Controllers;

// ─── Facade interface ─────────────────────────────────────────────────────────
// Hides all AiDotNet types from the controller; controller depends only on this.

public interface ISpamClassificationFacade
{
    Task<SpamClassificationResult> PredictAsync(
        IReadOnlyList<ClassificationSample> samples,
        CancellationToken ct = default);
}

// ─── Domain result ────────────────────────────────────────────────────────────
// What the facade returns — not an HTTP response type.

public sealed record SpamClassificationResult(
    List<SamplePrediction> Predictions,
    double TrainMs,
    double InferenceMs,
    string FacadePattern,
    string InterfaceChain);

// ─── Facade implementation ────────────────────────────────────────────────────
// Owns the AiDotNet model lifecycle: instantiate → train → predict.
// FTTransformerClassifier = Feature Tokenizer Transformer (per-feature embeddings →
// multi-head self-attention blocks × NumLayers → [CLS] token → linear classification head).

public sealed class TransformerSpamClassificationFacade : ISpamClassificationFacade
{
    public const int TrainSamples = 100;
    public const int TrainEpochs = 5;
    public const int EmbeddingDimension = 64;
    public const int NumHeads = 4;
    public const int NumLayers = 2;

    private const int FeatureCount = 8;
    private const int NumClasses = 2;
    private const double LearningRate = 0.001;

    // 100 deterministic samples (seed=42): 50 spam (label=1), 50 not-spam (label=0).
    private static readonly (double[,] Features, int[] Labels) TrainingData = GenerateTrainingData();

    public Task<SpamClassificationResult> PredictAsync(
        IReadOnlyList<ClassificationSample> samples,
        CancellationToken ct = default)
    {
        var trainTimer = Stopwatch.StartNew();

        // ─── 1. Build FTTransformerClassifier with a classification output head ───
        var options = new FTTransformerOptions<double>
        {
            EmbeddingDimension = EmbeddingDimension,
            NumHeads = NumHeads,
            NumLayers = NumLayers,
            DropoutRate = 0.1,
            NumFeatures = FeatureCount,
            EnableGradientClipping = true,
            MaxGradientNorm = 1.0,
            WeightDecay = 1e-5,
        };

        var classifier = new FTTransformerClassifier<double>(
            numNumericalFeatures: FeatureCount,
            numClasses: NumClasses,
            options: options);

        // ─── 2. Forward pass over the training corpus (architecture demonstration) ───
        // FTTransformerClassifier exposes TrainStep(Tensor, int[], lr, null) for full
        // gradient-based training. In AiDotNet 0.211.0 the backward pass through
        // MultiHeadAttentionLayer is not yet wired inside TrainStep (the UpdateParameters
        // guard throws "Backward pass must be called before updating parameters" on every
        // call regardless of state). Running Forward over the training batch proves the
        // transformer encoder computes without error and demonstrates the data path.
        // Parameter updates require the upcoming backward-pass fix in the library.
        var featuresTensor = BuildBatchTensor(TrainingData.Features, TrainSamples, FeatureCount);
        classifier.SetTrainingMode(true);

        for (int epoch = 0; epoch < TrainEpochs; epoch++)
            classifier.Forward(featuresTensor);   // forward pass; backward pending library fix

        trainTimer.Stop();

        // ─── 3. Inference: transformer forward → softmax → class probabilities ───
        var inferenceTimer = Stopwatch.StartNew();
        var predictions = new List<SamplePrediction>();
        classifier.SetTrainingMode(false);

        foreach (var sample in samples)
        {
            // PredictProbabilities returns Tensor<double> shape [1, NumClasses]
            var sampleTensor = new Tensor<double>(sample.Features.ToArray(), [1, FeatureCount]);
            var probs = classifier.PredictProbabilities(sampleTensor);

            // probs[0, 0] = P(not-spam)   probs[0, 1] = P(spam)
            double spamProb = probs.Rank >= 2 ? probs[0, 1] : probs[1];
            bool isSpam = spamProb >= 0.5;

            predictions.Add(new SamplePrediction(
                Features: sample.Features,
                Label: sample.Label,
                RawScore: Math.Round(spamProb, 6),
                Prediction: isSpam ? "spam" : "not-spam",
                Correct: sample.Label.HasValue
                    ? (sample.Label.Value == 1) == isSpam
                    : null));
        }

        inferenceTimer.Stop();

        return Task.FromResult(new SpamClassificationResult(
            Predictions: predictions,
            TrainMs: trainTimer.Elapsed.TotalMilliseconds,
            InferenceMs: inferenceTimer.Elapsed.TotalMilliseconds,
            FacadePattern:
                "ISpamClassificationFacade → TransformerSpamClassificationFacade" +
                " | new FTTransformerClassifier<double>(numNumericalFeatures=8, numClasses=2, FTTransformerOptions<double>)" +
                " → SetTrainingMode(true) → Forward×5(Tensor[100,8])" +
                " [TrainStep API: TrainStep(Tensor, int[], lr, null) — backward pass pending library fix]" +
                " → SetTrainingMode(false) → PredictProbabilities(Tensor[1,8])",
            InterfaceChain:
                "FTTransformerClassifier<T> → FTTransformerBase<T>"));
    }

    // ─── Helpers ─────────────────────────────────────────────────────────────

    private static Tensor<double> BuildBatchTensor(double[,] data, int rows, int cols)
    {
        var flat = new double[rows * cols];
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                flat[r * cols + c] = data[r, c];
        return new Tensor<double>(flat, [rows, cols]);
    }

    // ─── Training data ────────────────────────────────────────────────────────

    private static (double[,] Features, int[] Labels) GenerateTrainingData()
    {
        var rng = new Random(42);
        var features = new double[TrainSamples, FeatureCount];
        var labels = new int[TrainSamples];

        for (int i = 0; i < TrainSamples; i++)
        {
            bool isSpam = i < TrainSamples / 2;
            labels[i] = isSpam ? 1 : 0;

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
                features[i, 7] = rng.NextDouble() * 2.0 + 3.0;        // avgWordLen 3–5
            }
            else
            {
                // Legit: normal caps, few exclamations, no urgency patterns
                features[i, 0] = rng.Next(50, 300);                   // wordCount
                features[i, 1] = rng.NextDouble() * 0.1;              // capsRatio 0–0.1
                features[i, 2] = rng.Next(0, 2);                      // exclamations
                features[i, 3] = rng.Next(0, 2);                      // urlCount
                features[i, 4] = 0;                                   // moneyKeywords
                features[i, 5] = rng.Next(0, 1);                      // urgencyKeywords
                features[i, 6] = rng.NextDouble() * 0.05;             // linkRatio 0–0.05
                features[i, 7] = rng.NextDouble() * 2.5 + 5.0;        // avgWordLen 5–7.5
            }
        }

        return (features, labels);
    }
}

// ─── Controller ───────────────────────────────────────────────────────────────
// Thin HTTP layer: validates input, delegates to ISpamClassificationFacade, shapes response.

[ApiController]
[Route("api/[controller]")]
public sealed class ClassificationController : ControllerBase
{
    // Spam detection feature schema (8 features)
    // [0] wordCount  [1] capsRatio  [2] exclamations  [3] urlCount
    // [4] moneyKeywords  [5] urgencyKeywords  [6] linkRatio  [7] avgWordLen
    private const int FeatureCount = 8;

    private readonly ISpamClassificationFacade _facade;

    public ClassificationController(ISpamClassificationFacade facade) => _facade = facade;

    /// <summary>
    /// POC: Binary spam classification using FTTransformerClassifier (Feature Tokenizer Transformer
    /// + classification head) behind ISpamClassificationFacade — the proper application-level
    /// facade over AiDotNet's transformer API.
    /// Architecture: per-feature linear embeddings → NumLayers × multi-head self-attention →
    /// [CLS] linear head → softmax over {not-spam, spam}.
    /// </summary>
    [HttpPost("Predict")]
    public async Task<ActionResult<ClassificationPocResponse>> Predict(
        [FromBody] ClassificationRequest request,
        CancellationToken cancellationToken)
    {
        if (request.Samples is null || request.Samples.Count == 0)
            return BadRequest(new { error = "Provide at least one sample to classify." });

        foreach (var s in request.Samples)
        {
            if (s.Features.Count != FeatureCount)
                return BadRequest(new
                {
                    error = $"Each sample must have exactly {FeatureCount} features: " +
                            "wordCount, capsRatio, exclamations, urlCount, " +
                            "moneyKeywords, urgencyKeywords, linkRatio, avgWordLen"
                });
        }

        var totalTimer = Stopwatch.StartNew();
        var result = await _facade.PredictAsync(request.Samples, cancellationToken);
        totalTimer.Stop();

        double? accuracy = result.Predictions.All(p => p.Correct.HasValue)
            ? Math.Round(result.Predictions.Count(p => p.Correct == true) / (double)result.Predictions.Count, 4)
            : null;

        return Ok(new ClassificationPocResponse(
            Predictions: result.Predictions,
            ModelInfo: new ClassificationModelInfo(
                ModelType: "FTTransformerClassifier",
                EmbeddingDimension: TransformerSpamClassificationFacade.EmbeddingDimension,
                NumHeads: TransformerSpamClassificationFacade.NumHeads,
                NumLayers: TransformerSpamClassificationFacade.NumLayers,
                FeatureCount: FeatureCount,
                TrainSamples: TransformerSpamClassificationFacade.TrainSamples,
                TrainEpochs: TransformerSpamClassificationFacade.TrainEpochs,
                FeatureNames: ["wordCount", "capsRatio", "exclamations", "urlCount",
                               "moneyKeywords", "urgencyKeywords", "linkRatio", "avgWordLen"],
                ClassNames: ["not-spam", "spam"]),
            FacadePattern: result.FacadePattern,
            InterfaceChain: result.InterfaceChain,
            Accuracy: accuracy,
            Timings: new PocTimings(
                TotalMs: Math.Round(totalTimer.Elapsed.TotalMilliseconds, 3),
                TrainMs: Math.Round(result.TrainMs, 3),
                InferenceMs: Math.Round(result.InferenceMs, 3)),
            System: BuildSystemInfo()));
    }

    private static PocSystemInfo BuildSystemInfo() => new(
        Os: RuntimeInformation.OSDescription,
        Framework: RuntimeInformation.FrameworkDescription,
        LibraryVersion: typeof(FTTransformerClassifier<>).Assembly
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
    [property: JsonPropertyName("embeddingDimension")] int EmbeddingDimension,
    [property: JsonPropertyName("numHeads")] int NumHeads,
    [property: JsonPropertyName("numLayers")] int NumLayers,
    [property: JsonPropertyName("featureCount")] int FeatureCount,
    [property: JsonPropertyName("trainSamples")] int TrainSamples,
    [property: JsonPropertyName("trainEpochs")] int TrainEpochs,
    [property: JsonPropertyName("featureNames")] string[] FeatureNames,
    [property: JsonPropertyName("classNames")] string[] ClassNames);
