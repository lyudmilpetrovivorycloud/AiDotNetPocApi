using AiDotNet.ActivationFunctions;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
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
    int EpochsRun,
    long ParameterCount,
    string FacadePattern,
    string InterfaceChain);

// ─── Facade implementation ────────────────────────────────────────────────────
// Owns the AiDotNet model lifecycle. Registered as a SINGLETON: the model is
// trained exactly once (lazily, on first request, off the request thread) and
// reused for all subsequent inference.
//
// Model: FTTransformerNetwork<double> — canonical FT-Transformer (Gorishniy et al.
// 2021) assembled from the library's own layers via the custom-layers overload of
// NeuralNetworkArchitecture:
//
//   FeatureTokenizer(8 → 64) → PrependCLSToken → NumLayers × TransformerEncoder
//   → SequenceTokenSlice(First = CLS readout) → Dense(2, softmax)
//
// giving a proper per-sample [batch, NumClasses] output trained against plain
// one-hot targets with the network's default CategoricalCrossEntropyLoss.
//
// Why FTTransformerNetwork and not FTTransformerClassifier:
// In AiDotNet 0.213.3 the standalone tabular classifiers (FTTransformerClassifier,
// TabNetClassifier, AutoIntClassifier, …) all fail inside TrainStep with
// "Backward pass must be called before updating parameters" — verified empirically
// against the whole AiDotNet.NeuralNetworks.Tabular.*Classifier family.
// FTTransformerNetwork extends NeuralNetworkBase<T> and its standard Train(x, y)
// path (autodiff tape + optimizer) works: on this dataset loss converges
// 0.75 → <0.0005 within 30 epochs with 100% training accuracy.
//
// Why a custom layer stack: the network's DEFAULT stack has no CLS/pooling head —
// its dense head applies per feature token, so output is [batch, FeatureCount,
// NumClasses] and targets must be replicated per token. The custom stack above
// (all stock AiDotNet layers) restores the standard CLS readout.
//
// Also verified on 0.213.3: SaveModel/LoadModel does not round-trip this stack
// (reloaded network predicts garbage), so the model is trained per process rather
// than persisted; and layer instances must NOT be shared between two network
// instances (the architecture holds live, shape-bound layer objects).

public sealed class TransformerSpamClassificationFacade : ISpamClassificationFacade
{
    public const int TrainSamples = 100;
    public const int MaxTrainEpochs = 30;
    public const int EmbeddingDimension = 64;
    public const int NumHeads = 4;
    public const int NumLayers = 2;

    private const int FeatureCount = 8;
    private const int NumClasses = 2;
    private const int Seed = 42;

    // Stop training once mean cross-entropy drops below this (reached well before
    // MaxTrainEpochs on this dataset) — roughly a third off cold-start latency.
    private const double TargetLoss = 5e-4;

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
        int EpochsRun,
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
            // CLS-head output is [M, NumClasses] with rows summing to 1.
            double spamProb = probs[i, 1];
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
            EpochsRun: model.EpochsRun,
            ParameterCount: model.ParameterCount,
            FacadePattern:
                "ISpamClassificationFacade → TransformerSpamClassificationFacade (singleton, trained once)" +
                " | new FTTransformerNetwork<double>(NeuralNetworkArchitecture(custom layers:" +
                $" FeatureTokenizer(8→{EmbeddingDimension}) → PrependCLSToken → {NumLayers}× TransformerEncoder({NumHeads} heads, ff {EmbeddingDimension * 4})" +
                " → SequenceTokenSlice(CLS) → Dense(2, softmax)), FTTransformerOptions<double>)" +
                $" → SetTrainingMode(true) → Train(x[100,8], y[100,2] one-hot) until loss ≤ {TargetLoss} (max {MaxTrainEpochs} epochs, CategoricalCrossEntropyLoss)" +
                " → SetTrainingMode(false) → Predict(Tensor[M,8]) → softmax [M,2] → P(spam)",
            InterfaceChain:
                "FTTransformerNetwork<T> → NeuralNetworkBase<T> → INeuralNetworkModel<T>, IFullModel<T,Tensor<T>,Tensor<T>>");
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

        // Plain per-sample one-hot targets — the CLS head outputs [batch, NumClasses].
        var yFlat = new double[TrainSamples * NumClasses];
        for (int i = 0; i < TrainSamples; i++)
            yFlat[i * NumClasses + labels[i]] = 1.0;
        var y = new Tensor<double>(yFlat, [TrainSamples, NumClasses]);

        var network = BuildNetwork();

        network.SetTrainingMode(true);
        int epochsRun = 0;
        while (epochsRun < MaxTrainEpochs)
        {
            network.Train(x, y);
            epochsRun++;
            if (network.GetLastLoss() <= TargetLoss) break;
        }
        network.SetTrainingMode(false);

        timer.Stop();

        return new TrainedModel(
            Network: network,
            FeatureMeans: means,
            FeatureStds: stds,
            TrainMs: timer.Elapsed.TotalMilliseconds,
            FinalLoss: network.GetLastLoss(),
            EpochsRun: epochsRun,
            ParameterCount: network.GetParameterCount());
    }

    // Canonical FT-Transformer with CLS readout, assembled from stock AiDotNet
    // layers. Layer instances are live, shape-bound objects — never reuse this
    // list (or the architecture wrapping it) for a second network instance.
    private static FTTransformerNetwork<double> BuildNetwork()
    {
        var layers = new List<ILayer<double>>
        {
            new FeatureTokenizerLayer<double>(FeatureCount, EmbeddingDimension),
            new PrependCLSTokenLayer<double>(EmbeddingDimension, seed: Seed),
        };
        for (int i = 0; i < NumLayers; i++)
            layers.Add(new TransformerEncoderLayer<double>(
                NumHeads, feedForwardDim: EmbeddingDimension * 4, embeddingSize: EmbeddingDimension));
        layers.Add(new SequenceTokenSliceLayer<double>(SequenceTokenSliceLayer<double>.Position.First));
        layers.Add(new DenseLayer<double>(
            NumClasses, (IVectorActivationFunction<double>)new SoftmaxActivation<double>()));

        var architecture = new NeuralNetworkArchitecture<double>(
            InputType.OneDimensional,
            NeuralNetworkTaskType.MultiClassClassification,
            inputSize: FeatureCount,
            outputSize: NumClasses,
            layers: layers);

        var options = new FTTransformerOptions<double>
        {
            EmbeddingDimension = EmbeddingDimension,
            NumHeads = NumHeads,
            NumLayers = NumLayers,
            NumFeatures = FeatureCount,
            Seed = Seed,
        };

        return new FTTransformerNetwork<double>(architecture, options);
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
    /// NeuralNetworkBase.Train path (categorical cross-entropy, early-stopped at loss ≤ 5e-4)
    /// and cached; requests pay inference cost only. Architecture: per-feature linear embeddings
    /// → prepended CLS token → NumLayers × multi-head self-attention → CLS readout →
    /// softmax over {not-spam, spam}.
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
                TrainEpochs: result.EpochsRun,
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
