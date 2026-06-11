using AiDotNet.ActivationFunctions;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;
using Microsoft.AspNetCore.Mvc;
using System.Diagnostics;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Text.Json.Serialization;

namespace AiDotNetPocApi.Controllers;

// ─── Facade interface ─────────────────────────────────────────────────────────
// Hides all AiDotNet types from the controller; controller depends only on this.

public interface INerFacade
{
    Task<NerAnalysisResult> AnalyzeAsync(
        IReadOnlyList<string> sentences,
        CancellationToken ct = default);
}

// ─── Domain result ────────────────────────────────────────────────────────────

public sealed record NerAnalysisResult(
    List<SentenceResult> Results,
    double TrainMs,
    double InferenceMs,
    double FinalTrainLoss,
    int EpochsRun,
    long ParameterCount,
    string FacadePattern,
    string InterfaceChain);
    
// A neural network is a computer system loosely inspired by how the brain works.
// It's made of many small units called "neurons," organized in layers,
// where each connection between neurons has a weight
// (a number that says how strongly one neuron influences the next).
// You feed data in at one end — say, the 8 features of an email — and it passes through these layers,
// with each neuron combining its inputs, applying a simple math function,
// and passing the result forward until an answer comes out the other end
// (like "spam" or "not spam").
// The network "learns" by being shown many labeled examples
// and gradually adjusting all those weights so its answers get closer to the correct ones.
// After enough training, it can recognize patterns in new data it has never seen before.

// ─── Facade implementation ────────────────────────────────────────────────────
// Owns the AiDotNet model lifecycle. Registered as a SINGLETON: the model is
// trained exactly once (lazily, on first request, off the request thread) and
// reused for all subsequent inference.
//
// Model: NeuralNetwork<double> with a custom per-token transformer tagger stack
// built from stock AiDotNet layers:
//
//   Dense(50 → 64, per token) → 2× TransformerEncoder(4 heads, ff 256)
//   → Dense(9, softmax, per token)
//
// Input [batch, MaxSeqLen, 50] flows through unchanged in the sequence dims, so
// the output is a per-token label distribution [batch, MaxSeqLen, NumLabels],
// trained against per-token one-hot targets with CategoricalCrossEntropyLoss
// (the architecture default). Padding positions are labeled "O".
//
// Why not BiLSTMCRF / the AiModelBuilder facade (both verified on 0.213.3):
// - AiModelBuilder.BuildAsync() with DataLoaders.FromTensors over 5 sentences
//   throws "Validation tensors cannot have zero-sized dimensions" — the default
//   train/validation split produces an empty tensor on tiny datasets.
// - BiLSTMCRF's own Train(x, y) runs but does not learn: loss bottoms out
//   around epoch 50 then RISES (diverges) for every tried combination of
//   learning rate (0.01–0.5), dropout (0/default), hidden size (32/64), and
//   UseCRF on/off; token accuracy never beats the all-"O" majority baseline by
//   more than 2 tokens. The non-CRF path is flat at ln(9) — no learning at all.
// - This stock-layer stack on the standard NeuralNetworkBase.Train path (the
//   same path that fixed ClassificationController) reaches 100% token accuracy
//   on the training sentences with loss 3.36 → <0.01 in ~60 epochs.

public sealed class TransformerNerFacade : INerFacade
{
    public const int EmbeddingDim = 50;
    public const int ModelDim = 64;
    public const int NumHeads = 4;
    public const int NumEncoderLayers = 2;
    public const int NumLabels = 9;
    public const int MaxSeqLen = 15;
    public const int MaxTrainEpochs = 200;

    // Stop training once mean per-token cross-entropy drops below this
    // (reached around epoch 60 on this dataset; accuracy is 100% well before).
    private const double TargetLoss = 0.01;

    // CoNLL-2003 BIO label set (index → label name)
    public static readonly string[] LabelNames =
        ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"];

    // Hardcoded training sentences with ground-truth BIO label indices
    public static readonly (string[] Tokens, int[] Labels)[] TrainingSentences =
    [
        (
            ["Apple", "Inc", "announced", "the", "new", "iPhone", "in", "San", "Francisco"],
            [3, 4, 0, 0, 0, 0, 0, 5, 6]  // B-ORG I-ORG O O O O O B-LOC I-LOC
        ),
        (
            ["Elon", "Musk", "leads", "Tesla", "and", "SpaceX", "in", "California"],
            [1, 2, 0, 3, 0, 3, 0, 5]      // B-PER I-PER O B-ORG O B-ORG O B-LOC
        ),
        (
            ["Google", "acquired", "YouTube", "for", "over", "a", "billion", "dollars"],
            [3, 0, 7, 0, 0, 0, 0, 0]      // B-ORG O B-MISC O O O O O
        ),
        (
            ["Paris", "and", "London", "are", "popular", "European", "destinations"],
            [5, 0, 5, 0, 0, 0, 0]         // B-LOC O B-LOC O O O O
        ),
        (
            ["Barack", "Obama", "served", "as", "44th", "President"],
            [1, 2, 0, 0, 0, 0]            // B-PER I-PER O O O O
        ),
    ];

    // Trained once per process; all requests share the result.
    private readonly Lazy<Task<TrainedModel>> _model = new(
        () => Task.Run(TrainModel),
        LazyThreadSafetyMode.ExecutionAndPublication);

    // NeuralNetwork.Predict mutates internal layer state; serialize access.
    private readonly SemaphoreSlim _inferenceLock = new(1, 1);

    private sealed record TrainedModel(
        NeuralNetwork<double> Network,
        double TrainMs,
        double FinalLoss,
        int EpochsRun,
        long ParameterCount);

    public async Task<NerAnalysisResult> AnalyzeAsync(
        IReadOnlyList<string> sentences,
        CancellationToken ct = default)
    {
        // Training is shared state — never cancelled by an individual request;
        // the request just stops waiting for it.
        var model = await _model.Value.WaitAsync(ct);

        var inferenceTimer = Stopwatch.StartNew();

        // ─── Batched inference: one [M, MaxSeqLen, EmbeddingDim] tensor ───
        var tokenized = sentences.Select(Tokenize).ToArray();
        var batch = new double[sentences.Count * MaxSeqLen * EmbeddingDim];
        for (int i = 0; i < tokenized.Length; i++)
            WriteSentenceEmbeddings(tokenized[i], batch, i * MaxSeqLen * EmbeddingDim);

        Tensor<double> probs;
        await _inferenceLock.WaitAsync(ct);
        try
        {
            probs = model.Network.Predict(
                new Tensor<double>(batch, [sentences.Count, MaxSeqLen, EmbeddingDim]));
        }
        finally
        {
            _inferenceLock.Release();
        }

        var results = new List<SentenceResult>(sentences.Count);
        for (int i = 0; i < tokenized.Length; i++)
        {
            var tokens = tokenized[i];
            var bioLabels = DecodeBioLabels(probs, i, tokens.Length);
            results.Add(new SentenceResult(
                Sentence: sentences[i],
                Tokens: tokens,
                BioLabels: bioLabels,
                Entities: ExtractEntitySpans(tokens, bioLabels)));
        }

        inferenceTimer.Stop();

        return new NerAnalysisResult(
            Results: results,
            TrainMs: model.TrainMs,
            InferenceMs: inferenceTimer.Elapsed.TotalMilliseconds,
            FinalTrainLoss: model.FinalLoss,
            EpochsRun: model.EpochsRun,
            ParameterCount: model.ParameterCount,
            FacadePattern:
                "INerFacade → TransformerNerFacade (singleton, trained once)" +
                " | new NeuralNetwork<double>(NeuralNetworkArchitecture(custom layers:" +
                $" Dense({EmbeddingDim}→{ModelDim}) → {NumEncoderLayers}× TransformerEncoder({NumHeads} heads, ff {ModelDim * 4}) → Dense({NumLabels}, softmax)))" +
                $" → SetTrainingMode(true) → Train(x[{TrainingSentences.Length},{MaxSeqLen},{EmbeddingDim}], y[{TrainingSentences.Length},{MaxSeqLen},{NumLabels}] one-hot per token)" +
                $" until loss ≤ {TargetLoss} (max {MaxTrainEpochs} epochs, CategoricalCrossEntropyLoss)" +
                $" → SetTrainingMode(false) → Predict(Tensor[M,{MaxSeqLen},{EmbeddingDim}]) → per-token softmax [M,{MaxSeqLen},{NumLabels}] → argmax → BIO spans",
            InterfaceChain:
                "NeuralNetwork<T> → NeuralNetworkBase<T> → INeuralNetworkModel<T>, IFullModel<T,Tensor<T>,Tensor<T>>");
    }

    // Argmax over the label distribution per token. Tokens beyond MaxSeqLen are
    // outside the model window and reported as "O".
    private static string[] DecodeBioLabels(Tensor<double> probs, int sampleIndex, int tokenCount)
    {
        var labels = new string[tokenCount];
        for (int t = 0; t < tokenCount; t++)
        {
            if (t >= MaxSeqLen)
            {
                labels[t] = "O";
                continue;
            }
            int best = 0;
            double bestP = double.MinValue;
            for (int k = 0; k < NumLabels; k++)
            {
                if (probs[sampleIndex, t, k] > bestP)
                {
                    bestP = probs[sampleIndex, t, k];
                    best = k;
                }
            }
            labels[t] = LabelNames[best];
        }
        return labels;
    }

    // ─── One-time training ────────────────────────────────────────────────────

    private static TrainedModel TrainModel()
    {
        var timer = Stopwatch.StartNew();

        int batch = TrainingSentences.Length;
        var xFlat = new double[batch * MaxSeqLen * EmbeddingDim];
        var yFlat = new double[batch * MaxSeqLen * NumLabels];

        for (int b = 0; b < batch; b++)
        {
            var (tokens, labelIndices) = TrainingSentences[b];
            WriteSentenceEmbeddings(tokens, xFlat, b * MaxSeqLen * EmbeddingDim);
            for (int t = 0; t < MaxSeqLen; t++)
            {
                int label = t < labelIndices.Length ? labelIndices[t] : 0; // pad → O
                yFlat[(b * MaxSeqLen + t) * NumLabels + label] = 1.0;
            }
        }

        var x = new Tensor<double>(xFlat, [batch, MaxSeqLen, EmbeddingDim]);
        var y = new Tensor<double>(yFlat, [batch, MaxSeqLen, NumLabels]);

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
            TrainMs: timer.Elapsed.TotalMilliseconds,
            FinalLoss: network.GetLastLoss(),
            EpochsRun: epochsRun,
            ParameterCount: network.GetParameterCount());
    }

    // Per-token transformer tagger assembled from stock AiDotNet layers. The
    // dense layers map the last (embedding) dimension only, so sequence shape is
    // preserved end to end. Layer instances are live, shape-bound objects —
    // never reuse this list (or the architecture wrapping it) for a second
    // network instance.
    private static NeuralNetwork<double> BuildNetwork()
    {
        var layers = new List<ILayer<double>>
        {
            new DenseLayer<double>(ModelDim, (IActivationFunction<double>?)null),
        };
        for (int i = 0; i < NumEncoderLayers; i++)
            layers.Add(new TransformerEncoderLayer<double>(
                NumHeads, feedForwardDim: ModelDim * 4, embeddingSize: ModelDim));
        layers.Add(new DenseLayer<double>(
            NumLabels, (IVectorActivationFunction<double>)new SoftmaxActivation<double>()));

        var architecture = new NeuralNetworkArchitecture<double>(
            InputType.ThreeDimensional,
            NeuralNetworkTaskType.SequenceClassification,
            inputSize: EmbeddingDim,
            outputSize: NumLabels,
            layers: layers);

        return new NeuralNetwork<double>(architecture);
    }

    // ─── Embeddings / tokenization ────────────────────────────────────────────

    private static void WriteSentenceEmbeddings(string[] tokens, double[] target, int offset)
    {
        for (int t = 0; t < Math.Min(tokens.Length, MaxSeqLen); t++)
        {
            var emb = GetWordEmbedding(tokens[t]);
            Array.Copy(emb, 0, target, offset + t * EmbeddingDim, EmbeddingDim);
        }
    }

    // Deterministic hash-based word embedding for POC (not semantically meaningful,
    // but consistent — same word always maps to the same 50-dim vector)
    private static double[] GetWordEmbedding(string word)
    {
        var hash = Math.Abs(word.ToLowerInvariant().GetHashCode());
        var rng = new Random(hash);
        var emb = new double[EmbeddingDim];
        for (int i = 0; i < EmbeddingDim; i++)
            emb[i] = rng.NextDouble() * 2.0 - 1.0;
        return emb;
    }

    private static string[] Tokenize(string sentence) =>
        sentence.Split(' ', StringSplitOptions.RemoveEmptyEntries);

    // ─── BIO span extraction ──────────────────────────────────────────────────

    private static List<EntitySpan> ExtractEntitySpans(string[] tokens, string[] bioLabels)
    {
        var entities = new List<EntitySpan>();
        int i = 0;
        while (i < tokens.Length)
        {
            var label = bioLabels[i];
            if (label.StartsWith("B-", StringComparison.Ordinal))
            {
                var entityType = label[2..];
                var spanTokens = new List<string> { tokens[i] };
                int start = i++;

                while (i < tokens.Length && bioLabels[i] == $"I-{entityType}")
                {
                    spanTokens.Add(tokens[i++]);
                }

                entities.Add(new EntitySpan(
                    Text: string.Join(" ", spanTokens),
                    Type: entityType,
                    StartToken: start,
                    EndToken: i - 1));
            }
            else
            {
                i++;
            }
        }
        return entities;
    }
}

// ─── Controller ───────────────────────────────────────────────────────────────
// Thin HTTP layer: validates input, delegates to INerFacade, shapes response.

[ApiController]
[Route("api/[controller]")]
public sealed class NerController : ControllerBase
{
    private readonly INerFacade _facade;

    public NerController(INerFacade facade) => _facade = facade;

    /// <summary>
    /// POC: Named Entity Recognition using a per-token transformer tagger built from
    /// stock AiDotNet layers behind INerFacade. The model is trained once per process
    /// via the standard NeuralNetworkBase.Train path (categorical cross-entropy,
    /// early-stopped at loss ≤ 0.01) and cached; requests pay inference cost only.
    /// Architecture: per-token Dense(50→64) → 2× multi-head self-attention →
    /// per-token softmax over the 9 CoNLL-2003 BIO labels → entity span extraction.
    /// </summary>
    [HttpPost("Analyze")]
    public async Task<ActionResult<NerPocResponse>> Analyze(
        [FromBody] NerRequest request,
        CancellationToken cancellationToken)
    {
        if (request.Sentences is null || request.Sentences.Count == 0)
            return BadRequest(new { error = "Provide at least one sentence to analyze." });

        for (int i = 0; i < request.Sentences.Count; i++)
        {
            if (string.IsNullOrWhiteSpace(request.Sentences[i]))
                return BadRequest(new { error = $"Sentence {i}: must be a non-empty string." });
        }

        var totalTimer = Stopwatch.StartNew();
        var result = await _facade.AnalyzeAsync(request.Sentences, cancellationToken);
        totalTimer.Stop();

        return Ok(new NerPocResponse(
            Results: result.Results,
            ModelInfo: new NerModelInfo(
                ModelType: "TransformerTokenClassifier",
                EmbeddingDim: TransformerNerFacade.EmbeddingDim,
                ModelDim: TransformerNerFacade.ModelDim,
                NumHeads: TransformerNerFacade.NumHeads,
                NumLayers: TransformerNerFacade.NumEncoderLayers,
                NumLabels: TransformerNerFacade.NumLabels,
                MaxSeqLen: TransformerNerFacade.MaxSeqLen,
                TrainingSentences: TransformerNerFacade.TrainingSentences.Length,
                TrainEpochs: result.EpochsRun,
                ParameterCount: result.ParameterCount,
                FinalTrainLoss: Math.Round(result.FinalTrainLoss, 6),
                LabelSet: TransformerNerFacade.LabelNames),
            FacadePattern: result.FacadePattern,
            InterfaceChain: result.InterfaceChain,
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
        LibraryVersion: typeof(NeuralNetwork<>).Assembly
            .GetCustomAttribute<AssemblyInformationalVersionAttribute>()?.InformationalVersion
            ?? "unknown");
}

// ─── Request / Response contracts ────────────────────────────────────────────

public sealed record NerRequest(
    [property: JsonPropertyName("sentences")] List<string> Sentences);

public sealed record NerPocResponse(
    [property: JsonPropertyName("results")] List<SentenceResult> Results,
    [property: JsonPropertyName("modelInfo")] NerModelInfo ModelInfo,
    [property: JsonPropertyName("facadePattern")] string FacadePattern,
    [property: JsonPropertyName("interfaceChain")] string InterfaceChain,
    [property: JsonPropertyName("timings")] PocTimings Timings,
    [property: JsonPropertyName("system")] PocSystemInfo System);

public sealed record SentenceResult(
    [property: JsonPropertyName("sentence")] string Sentence,
    [property: JsonPropertyName("tokens")] string[] Tokens,
    [property: JsonPropertyName("bioLabels")] string[] BioLabels,
    [property: JsonPropertyName("entities")] List<EntitySpan> Entities);

public sealed record EntitySpan(
    [property: JsonPropertyName("text")] string Text,
    [property: JsonPropertyName("type")] string Type,
    [property: JsonPropertyName("startToken")] int StartToken,
    [property: JsonPropertyName("endToken")] int EndToken);

public sealed record NerModelInfo(
    [property: JsonPropertyName("modelType")] string ModelType,
    [property: JsonPropertyName("embeddingDim")] int EmbeddingDim,
    [property: JsonPropertyName("modelDim")] int ModelDim,
    [property: JsonPropertyName("numHeads")] int NumHeads,
    [property: JsonPropertyName("numLayers")] int NumLayers,
    [property: JsonPropertyName("numLabels")] int NumLabels,
    [property: JsonPropertyName("maxSeqLen")] int MaxSeqLen,
    [property: JsonPropertyName("trainingSentences")] int TrainingSentences,
    [property: JsonPropertyName("trainEpochs")] int TrainEpochs,
    [property: JsonPropertyName("parameterCount")] long ParameterCount,
    [property: JsonPropertyName("finalTrainLoss")] double FinalTrainLoss,
    [property: JsonPropertyName("labelSet")] string[] LabelSet);

public sealed record PocTimings(
    [property: JsonPropertyName("totalMs")] double TotalMs,
    [property: JsonPropertyName("trainMs")] double TrainMs,
    [property: JsonPropertyName("inferenceMs")] double InferenceMs);

public sealed record PocSystemInfo(
    [property: JsonPropertyName("os")] string Os,
    [property: JsonPropertyName("framework")] string Framework,
    [property: JsonPropertyName("libraryVersion")] string LibraryVersion);
