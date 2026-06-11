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
using System.Text.RegularExpressions;

namespace AiDotNetPocApi.Controllers;

// Classification is a type of machine learning task where the goal is to sort things into categories.
// You give the model an example — like an email 
// and its features — and it predicts which group that example belongs to from a fixed set of labels,
// such as "spam" or "not spam."
// The model learns by studying lots of examples where the correct category is already known,
// picking up on which patterns tend to go with which label.
// Once trained, it can take a brand-new example and assign it to the most likely category.
// When there are just two possible labels (spam vs. not spam)
// it's called binary classification; when there are more
// (say, sorting email into "work," "personal," "promotions," and "spam")
// it's called multi-class classification.

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

// ─── Feature extraction ───────────────────────────────────────────────────────
// Derives the model's 8-feature vector from raw message text, mirroring the
// semantics the synthetic training data was generated under. This is the bridge
// that lets the API accept real email text instead of pre-computed features.
//
// Definitions (must stay consistent with the training-data generator above —
// the model only knows these features in these units):
//   [0] wordCount        whitespace-separated tokens
//   [1] capsRatio        uppercase letters / total letters (0 when no letters)
//   [2] exclamations     count of '!' characters
//   [3] urlCount         matches of http(s):// or www. URLs
//   [4] moneyKeywords    occurrences of money patterns (free, $$$, winner,
//                        prize, % off, cash, win)
//   [5] urgencyKeywords  occurrences of urgency patterns (act now, urgent,
//                        expires/expiring, limited, hurry, last chance)
//   [6] linkRatio        URL tokens / wordCount (0–1)
//   [7] avgWordLen       mean letter+digit count per non-URL token

public static class SpamFeatureExtractor
{
    private static readonly Regex UrlPattern = new(
        @"(https?://|www\.)\S+", RegexOptions.IgnoreCase | RegexOptions.Compiled);

    private static readonly Regex[] MoneyPatterns = BuildPatterns(
        @"\bfree\b", @"\$\$\$", @"\bwinner\b", @"\bprize\b", @"%\s*off",
        @"\bcash\b", @"\bwin\b");

    private static readonly Regex[] UrgencyPatterns = BuildPatterns(
        @"\bact\s+now\b", @"\burgent\w*\b", @"\bexpir\w+\b", @"\blimited\b",
        @"\bhurry\b", @"\blast\s+chance\b");

    private static Regex[] BuildPatterns(params string[] patterns) =>
        [.. patterns.Select(p => new Regex(p, RegexOptions.IgnoreCase | RegexOptions.Compiled))];

    public static List<double> Extract(string text)
    {
        var tokens = text.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries);
        int wordCount = tokens.Length;

        int letters = 0, uppercase = 0, exclamations = 0;
        foreach (char c in text)
        {
            if (char.IsLetter(c))
            {
                letters++;
                if (char.IsUpper(c)) uppercase++;
            }
            else if (c == '!')
            {
                exclamations++;
            }
        }

        int urlCount = UrlPattern.Matches(text).Count;
        int urlTokens = tokens.Count(t => UrlPattern.IsMatch(t));
        int moneyKeywords = MoneyPatterns.Sum(p => p.Matches(text).Count);
        int urgencyKeywords = UrgencyPatterns.Sum(p => p.Matches(text).Count);

        // URL tokens are excluded so a single 40-char link doesn't dominate the
        // mean; token "length" counts letters/digits only, so "WIN!!!" measures 3.
        double totalLen = 0;
        int measured = 0;
        foreach (var token in tokens)
        {
            if (UrlPattern.IsMatch(token)) continue;
            int len = token.Count(char.IsLetterOrDigit);
            if (len == 0) continue;
            totalLen += len;
            measured++;
        }

        return
        [
            wordCount,
            Math.Round(letters > 0 ? (double)uppercase / letters : 0.0, 4),
            exclamations,
            urlCount,
            moneyKeywords,
            urgencyKeywords,
            Math.Round(wordCount > 0 ? (double)urlTokens / wordCount : 0.0, 4),
            Math.Round(measured > 0 ? totalLen / measured : 0.0, 4),
        ];
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

        return Ok(new ClassificationPocResponse(
            Predictions: result.Predictions,
            ModelInfo: BuildModelInfo(result),
            FacadePattern: result.FacadePattern,
            InterfaceChain: result.InterfaceChain,
            Accuracy: ComputeAccuracy(result.Predictions),
            Timings: BuildTimings(totalTimer, result),
            System: BuildSystemInfo()));
    }

    /// <summary>
    /// POC: classify raw email/message text. SpamFeatureExtractor derives the same
    /// 8-feature vector the model was trained on directly from each text, and the
    /// samples are then served by the same trained FTTransformerNetwork as Predict.
    /// Each prediction echoes the extracted features so the derivation is auditable.
    /// </summary>
    [HttpPost("PredictText")]
    public async Task<ActionResult<TextClassificationPocResponse>> PredictText(
        [FromBody] TextClassificationRequest request,
        CancellationToken cancellationToken)
    {
        if (request.Messages is null || request.Messages.Count == 0)
            return BadRequest(new { error = "Provide at least one message to classify." });

        for (int i = 0; i < request.Messages.Count; i++)
        {
            if (string.IsNullOrWhiteSpace(request.Messages[i].Text))
                return BadRequest(new { error = $"Message {i}: text must be a non-empty string." });
        }

        var totalTimer = Stopwatch.StartNew();

        var samples = request.Messages
            .Select(m => new ClassificationSample(SpamFeatureExtractor.Extract(m.Text), m.Label))
            .ToList();

        var result = await _facade.PredictAsync(samples, cancellationToken);
        totalTimer.Stop();

        var predictions = result.Predictions
            .Select((p, i) => new TextSamplePrediction(
                Text: request.Messages[i].Text,
                Features: p.Features,
                Label: p.Label,
                RawScore: p.RawScore,
                Prediction: p.Prediction,
                Correct: p.Correct))
            .ToList();

        return Ok(new TextClassificationPocResponse(
            Predictions: predictions,
            ModelInfo: BuildModelInfo(result),
            FacadePattern: result.FacadePattern,
            InterfaceChain: result.InterfaceChain,
            Accuracy: ComputeAccuracy(result.Predictions),
            Timings: BuildTimings(totalTimer, result),
            System: BuildSystemInfo()));
    }

    private static double? ComputeAccuracy(List<SamplePrediction> predictions) =>
        predictions.All(p => p.Correct.HasValue)
            ? Math.Round(predictions.Count(p => p.Correct == true) / (double)predictions.Count, 4)
            : null;

    private static ClassificationModelInfo BuildModelInfo(SpamClassificationResult result) => new(
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
        ClassNames: ["not-spam", "spam"]);

    private static PocTimings BuildTimings(Stopwatch totalTimer, SpamClassificationResult result) => new(
        TotalMs: Math.Round(totalTimer.Elapsed.TotalMilliseconds, 3),
        TrainMs: Math.Round(result.TrainMs, 3),
        InferenceMs: Math.Round(result.InferenceMs, 3));

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

public sealed record TextClassificationRequest(
    [property: JsonPropertyName("messages")] List<TextSample> Messages);

public sealed record TextSample(
    [property: JsonPropertyName("text")] string Text,
    [property: JsonPropertyName("label")] int? Label);

public sealed record TextClassificationPocResponse(
    [property: JsonPropertyName("predictions")] List<TextSamplePrediction> Predictions,
    [property: JsonPropertyName("modelInfo")] ClassificationModelInfo ModelInfo,
    [property: JsonPropertyName("facadePattern")] string FacadePattern,
    [property: JsonPropertyName("interfaceChain")] string InterfaceChain,
    [property: JsonPropertyName("accuracy")] double? Accuracy,
    [property: JsonPropertyName("timings")] PocTimings Timings,
    [property: JsonPropertyName("system")] PocSystemInfo System);

public sealed record TextSamplePrediction(
    [property: JsonPropertyName("text")] string Text,
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
