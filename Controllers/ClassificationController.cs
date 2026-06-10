using AiDotNet.Enums;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Tabular;
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
    double FinalTrainLoss,
    long ParameterCount,
    string FacadePattern,
    string InterfaceChain);

// ─── Facade implementation ────────────────────────────────────────────────────
// Owns the AiDotNet model lifecycle. Registered as a SINGLETON: the model is
// trained exactly once (lazily, on first request, off the request thread) and
// reused for all subsequent inference.
//
// Model: FTTransformerNetwork<double> — Feature Tokenizer Transformer (per-feature
// linear embeddings → NumLayers × multi-head self-attention → per-token softmax head).
//
// Why FTTransformerNetwork and not FTTransformerClassifier:
// In AiDotNet 0.213.3 the standalone tabular classifiers (FTTransformerClassifier,
// TabNetClassifier, AutoIntClassifier, …) all fail inside TrainStep with
// "Backward pass must be called before updating parameters" — verified empirically
// against the whole AiDotNet.NeuralNetworks.Tabular.*Classifier family.
// FTTransformerNetwork extends NeuralNetworkBase<T> and its standard Train(x, y)
// path (autodiff tape + optimizer) works: on this dataset loss converges
// 0.73 → <0.001 over 30 epochs with 100% training accuracy.
//
// One library quirk handled here: the default FT-Transformer layer stack has no
// final [CLS]/pooling layer, so the network outputs per-feature-token predictions
// of shape [batch, FeatureCount, NumClasses]. We therefore replicate the one-hot
// target across the 8 feature tokens during training and mean-pool the per-token
// softmax probabilities at inference.

public sealed class TransformerSpamClassificationFacade : ISpamClassificationFacade
{
    public const int TrainSamples = 100;
    public const int TrainEpochs = 30;
    public const int EmbeddingDimension = 64;
    public const int NumHeads = 4;
    public const int NumLayers = 2;

    private const int FeatureCount = 8;
    private const int NumClasses = 2;

    // Trained once per process; all requests share the result.
    private readonly Lazy<Task<TrainedModel>> _model = new(
        () => Task.Run(TrainModel),
        LazyThreadSafetyMode.ExecutionAndPublication);

    // FTTransformerNetwork.Predict mutates internal layer state; serialize access.
    private readonly SemaphoreSlim _inferenceLock = new(1, 1);

    private sealed record TrainedModel(
        FTTransformerNetwork<double> Network,
        double[] FeatureMeans,
        double[] FeatureStds,
        double TrainMs,
        double FinalLoss,
        long ParameterCount);

    public async Task<SpamClassificationResult> PredictAsync(
        IReadOnlyList<ClassificationSample> samples,
        CancellationToken ct = default)
    {
        // Training is shared state — never cancelled by an individual request;
        // the request just stops waiting for it.
        var model = await _model.Value.WaitAsync(ct);

        var inferenceTimer = Stopwatch.StartNew();

        // ─── Batched inference: one [M, 8] tensor for all samples ───
        var batch = new double[samples.Count * FeatureCount];
        for (int i = 0; i < samples.Count; i++)
            for (int c = 0; c < FeatureCount; c++)
                batch[i * FeatureCount + c] =
                    (samples[i].Features[c] - model.FeatureMeans[c]) / model.FeatureStds[c];

        Tensor<double> probs;
        await _inferenceLock.WaitAsync(ct);
        try
        {
            probs = model.Network.Predict(
                new Tensor<double>(batch, [samples.Count, FeatureCount]));
        }
        finally
        {
            _inferenceLock.Release();
        }

        var predictions = new List<SamplePrediction>(samples.Count);
        for (int i = 0; i < samples.Count; i++)
        {
            double spamProb = MeanPooledSpamProbability(probs, i);
            bool isSpam = spamProb >= 0.5;
            var sample = samples[i];

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

        return new SpamClassificationResult(
            Predictions: predictions,
            TrainMs: model.TrainMs,
            InferenceMs: inferenceTimer.Elapsed.TotalMilliseconds,
            FinalTrainLoss: model.FinalLoss,
            ParameterCount: model.ParameterCount,
            FacadePattern:
                "ISpamClassificationFacade → TransformerSpamClassificationFacade (singleton, trained once)" +
                " | new FTTransformerNetwork<double>(NeuralNetworkArchitecture(OneDimensional, MultiClassClassification, inputSize=8, outputSize=2), FTTransformerOptions<double>)" +
                $" → SetTrainingMode(true) → Train×{TrainEpochs}(x[100,8], y[100,8,2] one-hot per token)" +
                " → SetTrainingMode(false) → Predict(Tensor[M,8]) → mean-pool per-token softmax [M,8,2] → P(spam)",
            InterfaceChain:
                "FTTransformerNetwork<T> → NeuralNetworkBase<T> → INeuralNetworkModel<T>, IFullModel<T,Tensor<T>,Tensor<T>>");
    }

    // Network output is [M, FeatureCount, NumClasses] (per-token softmax, rows sum
    // to 1); average P(class=1) over the feature tokens. Falls back to [M, NumClasses]
    // in case a future library version adds the pooling head.
    private static double MeanPooledSpamProbability(Tensor<double> probs, int sampleIndex)
    {
        if (probs.Rank == 3)
        {
            double sum = 0;
            int tokens = probs.Shape[1];
            for (int t = 0; t < tokens; t++)
                sum += probs[sampleIndex, t, 1];
            return sum / tokens;
        }
        return probs[sampleIndex, 1];
    }

    // ─── One-time training ────────────────────────────────────────────────────

    private static TrainedModel TrainModel()
    {
        var timer = Stopwatch.StartNew();

        var (features, labels) = GenerateTrainingData();
        var (means, stds) = ComputeNormalization(features);

        // z-score normalize: raw feature scales differ by 3 orders of magnitude
        // (wordCount up to 300 vs linkRatio 0–0.5) — unnormalized input keeps the
        // transformer from converging.
        var xFlat = new double[TrainSamples * FeatureCount];
        for (int r = 0; r < TrainSamples; r++)
            for (int c = 0; c < FeatureCount; c++)
                xFlat[r * FeatureCount + c] = (features[r, c] - means[c]) / stds[c];
        var x = new Tensor<double>(xFlat, [TrainSamples, FeatureCount]);

        // One-hot targets replicated across the 8 feature tokens to match the
        // network's per-token output shape [batch, FeatureCount, NumClasses].
        var yFlat = new double[TrainSamples * FeatureCount * NumClasses];
        for (int i = 0; i < TrainSamples; i++)
            for (int t = 0; t < FeatureCount; t++)
                yFlat[(i * FeatureCount + t) * NumClasses + labels[i]] = 1.0;
        var y = new Tensor<double>(yFlat, [TrainSamples, FeatureCount, NumClasses]);

        var architecture = new NeuralNetworkArchitecture<double>(
            InputType.OneDimensional,
            NeuralNetworkTaskType.MultiClassClassification,
            inputSize: FeatureCount,
            outputSize: NumClasses);

        var options = new FTTransformerOptions<double>
        {
            EmbeddingDimension = EmbeddingDimension,
            NumHeads = NumHeads,
            NumLayers = NumLayers,
            DropoutRate = 0.1,
            NumFeatures = FeatureCount,
        };

        var network = new FTTransformerNetwork<double>(architecture, options);

        network.SetTrainingMode(true);
        for (int epoch = 0; epoch < TrainEpochs; epoch++)
            network.Train(x, y);
        network.SetTrainingMode(false);

        timer.Stop();

        return new TrainedModel(
            Network: network,
            FeatureMeans: means,
            FeatureStds: stds,
            TrainMs: timer.Elapsed.TotalMilliseconds,
            FinalLoss: network.GetLastLoss(),
            ParameterCount: network.GetParameterCount());
    }

    private static (double[] Means, double[] Stds) ComputeNormalization(double[,] features)
    {
        var means = new double[FeatureCount];
        var stds = new double[FeatureCount];
        for (int c = 0; c < FeatureCount; c++)
        {
            for (int r = 0; r < TrainSamples; r++) means[c] += features[r, c];
            means[c] /= TrainSamples;
            for (int r = 0; r < TrainSamples; r++) stds[c] += Math.Pow(features[r, c] - means[c], 2);
            stds[c] = Math.Sqrt(stds[c] / TrainSamples);
            if (stds[c] < 1e-12) stds[c] = 1.0;
        }
        return (means, stds);
    }

    // ─── Training data ────────────────────────────────────────────────────────
    // 100 deterministic samples (seed=42): 50 spam (label=1), 50 not-spam (label=0).

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
    /// POC: Binary spam classification using FTTransformerNetwork (Feature Tokenizer Transformer)
    /// behind ISpamClassificationFacade. The model is trained once per process via the standard
    /// NeuralNetworkBase.Train path (loss converges to &lt;0.001) and cached; requests pay
    /// inference cost only. Architecture: per-feature linear embeddings → NumLayers ×
    /// multi-head self-attention → per-token softmax, mean-pooled over {not-spam, spam}.
    /// </summary>
    [HttpPost("Predict")]
    public async Task<ActionResult<ClassificationPocResponse>> Predict(
        [FromBody] ClassificationRequest request,
        CancellationToken cancellationToken)
    {
        if (request.Samples is null || request.Samples.Count == 0)
            return BadRequest(new { error = "Provide at least one sample to classify." });

        for (int i = 0; i < request.Samples.Count; i++)
        {
            var s = request.Samples[i];
            if (s.Features is null || s.Features.Count != FeatureCount)
                return BadRequest(new
                {
                    error = $"Sample {i}: each sample must have exactly {FeatureCount} features: " +
                            "wordCount, capsRatio, exclamations, urlCount, " +
                            "moneyKeywords, urgencyKeywords, linkRatio, avgWordLen"
                });
            if (s.Features.Any(f => !double.IsFinite(f)))
                return BadRequest(new { error = $"Sample {i}: features must be finite numbers." });
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
                ModelType: "FTTransformerNetwork",
                EmbeddingDimension: TransformerSpamClassificationFacade.EmbeddingDimension,
                NumHeads: TransformerSpamClassificationFacade.NumHeads,
                NumLayers: TransformerSpamClassificationFacade.NumLayers,
                FeatureCount: FeatureCount,
                TrainSamples: TransformerSpamClassificationFacade.TrainSamples,
                TrainEpochs: TransformerSpamClassificationFacade.TrainEpochs,
                ParameterCount: result.ParameterCount,
                FinalTrainLoss: Math.Round(result.FinalTrainLoss, 6),
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
        // NB: AiDotNet 0.213.3 ships with stale assembly version attributes (0.204.0);
        // this reports what the loaded assembly declares about itself.
        LibraryVersion: typeof(FTTransformerNetwork<>).Assembly
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
    [property: JsonPropertyName("parameterCount")] long ParameterCount,
    [property: JsonPropertyName("finalTrainLoss")] double FinalTrainLoss,
    [property: JsonPropertyName("featureNames")] string[] FeatureNames,
    [property: JsonPropertyName("classNames")] string[] ClassNames);
