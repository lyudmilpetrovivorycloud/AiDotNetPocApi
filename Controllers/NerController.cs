using AiDotNet;
using AiDotNet.Data.Loaders;
using AiDotNet.NER.Options;
using AiDotNet.NER.SequenceLabeling;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.LinearAlgebra;
using Microsoft.AspNetCore.Mvc;
using System.Diagnostics;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Text.Json.Serialization;

namespace AiDotNetPocApi.Controllers;

[ApiController]
[Route("api/[controller]")]
public sealed class NerController : ControllerBase
{
    private const int EmbeddingDim = 50;
    private const int HiddenDim = 64;
    private const int NumLabels = 9;
    private const int MaxSeqLen = 15;

    // CoNLL-2003 BIO label set (index → label name)
    private static readonly string[] LabelNames =
        ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"];

    // Hardcoded training sentences with ground-truth BIO label indices
    private static readonly (string[] Tokens, int[] Labels)[] TrainingSentences =
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

    /// <summary>
    /// POC: Named Entity Recognition using BiLSTMCRF via AiDotNet facade.
    /// Demonstrates DataLoaders.FromTensors → AiModelBuilder.ConfigureModel → BuildAsync → Predict.
    /// </summary>
    [HttpPost("Analyze")]
    public async Task<ActionResult<NerPocResponse>> Analyze(
        [FromBody] NerRequest request,
        CancellationToken cancellationToken)
    {
        if (request.Sentences is null || request.Sentences.Count == 0)
            return BadRequest(new { error = "Provide at least one sentence to analyze." });

        var totalTimer = Stopwatch.StartNew();
        var trainTimer = Stopwatch.StartNew();

        // ─────────────────────────────────────────────────────────────────
        // 1. Build training tensors from hardcoded labeled sentences
        // ─────────────────────────────────────────────────────────────────
        var (trainEmbeddings, trainLabels) = BuildTrainingTensors();

        // ─────────────────────────────────────────────────────────────────
        // 2. Wrap training data in AiDotNet's tensor data loader
        // ─────────────────────────────────────────────────────────────────
        var dataLoader = DataLoaders.FromTensors(trainEmbeddings, trainLabels);

        // ─────────────────────────────────────────────────────────────────
        // 3. Configure BiLSTMCRF via AiModelBuilder facade
        //    BiLSTMCRF implements INERModel<T> : IFullModel<T, Tensor<T>, Tensor<T>>
        //    so AiModelBuilder<double, Tensor<double>, Tensor<double>> accepts it directly.
        // ─────────────────────────────────────────────────────────────────
        var architecture = new NeuralNetworkArchitecture<double>(
            inputFeatures: EmbeddingDim,
            outputSize: NumLabels);

        var nerOptions = new BiLSTMCRFOptions
        {
            EmbeddingDimension = EmbeddingDim,
            HiddenDimension = HiddenDim,
            NumLabels = NumLabels,
            MaxSequenceLength = MaxSeqLen,
            UseCRF = true,
        };

        var nerModel = new BiLSTMCRF<double>(architecture, nerOptions);

        // Full facade path: ConfigureDataLoader + ConfigureModel + BuildAsync
        var result = await new AiModelBuilder<double, Tensor<double>, Tensor<double>>()
            .ConfigureDataLoader(dataLoader)
            .ConfigureModel(nerModel)
            .BuildAsync();

        trainTimer.Stop();

        // ─────────────────────────────────────────────────────────────────
        // 4. Run entity recognition on each request sentence (AiModelResult.Predict)
        // ─────────────────────────────────────────────────────────────────
        var inferenceTimer = Stopwatch.StartNew();
        var sentenceResults = new List<SentenceResult>();

        foreach (var sentence in request.Sentences)
        {
            var tokens = Tokenize(sentence);
            var testTensor = BuildSentenceTensor(tokens);

            // AiModelResult<double, Tensor<double>, Tensor<double>>.Predict uses CRF Viterbi decoding
            var labelTensor = result.Predict(testTensor);
            var bioLabels = DecodeLabelTensor(labelTensor, tokens.Length);
            var entities = ExtractEntitySpans(tokens, bioLabels);

            sentenceResults.Add(new SentenceResult(
                Sentence: sentence,
                Tokens: tokens,
                BioLabels: bioLabels,
                Entities: entities));
        }

        inferenceTimer.Stop();
        totalTimer.Stop();

        var facadePattern = "AiModelBuilder<double, Tensor<double>, Tensor<double>>" +
                            ".ConfigureDataLoader(DataLoaders.FromTensors(...))" +
                            $".ConfigureModel({nameof(BiLSTMCRF<double>)}<double>)" +
                            ".BuildAsync() → AiModelResult.Predict(Tensor<double>)";

        return Ok(new NerPocResponse(
            Results: sentenceResults,
            ModelInfo: new NerModelInfo(
                ModelType: "BiLSTMCRF",
                EmbeddingDim: EmbeddingDim,
                HiddenDim: HiddenDim,
                NumLabels: NumLabels,
                UseCRF: true,
                TrainingSentences: TrainingSentences.Length,
                LabelSet: LabelNames),
            FacadePattern: facadePattern,
            InterfaceChain: "BiLSTMCRF → INERModel<T> → IFullModel<T, Tensor<T>, Tensor<T>>",
            Timings: new PocTimings(
                TotalMs: Math.Round(totalTimer.Elapsed.TotalMilliseconds, 3),
                TrainMs: Math.Round(trainTimer.Elapsed.TotalMilliseconds, 3),
                InferenceMs: Math.Round(inferenceTimer.Elapsed.TotalMilliseconds, 3)),
            System: BuildSystemInfo()));
    }

    // ─── Tensor construction ─────────────────────────────────────────────

    private static (Tensor<double> embeddings, Tensor<double> labels) BuildTrainingTensors()
    {
        int batch = TrainingSentences.Length;

        // [batch, MaxSeqLen, EmbeddingDim]
        var embValues = new double[batch * MaxSeqLen * EmbeddingDim];
        // [batch, MaxSeqLen]
        var lblValues = new double[batch * MaxSeqLen];

        for (int b = 0; b < batch; b++)
        {
            var (tokens, labelIndices) = TrainingSentences[b];
            for (int t = 0; t < Math.Min(tokens.Length, MaxSeqLen); t++)
            {
                var emb = GetWordEmbedding(tokens[t]);
                for (int d = 0; d < EmbeddingDim; d++)
                    embValues[b * MaxSeqLen * EmbeddingDim + t * EmbeddingDim + d] = emb[d];

                lblValues[b * MaxSeqLen + t] = t < labelIndices.Length ? labelIndices[t] : 0;
            }
        }

        return (
            new Tensor<double>(embValues, [batch, MaxSeqLen, EmbeddingDim]),
            new Tensor<double>(lblValues, [batch, MaxSeqLen]));
    }

    private static Tensor<double> BuildSentenceTensor(string[] tokens)
    {
        // [1, MaxSeqLen, EmbeddingDim] — single sentence, padded
        var values = new double[1 * MaxSeqLen * EmbeddingDim];
        for (int t = 0; t < Math.Min(tokens.Length, MaxSeqLen); t++)
        {
            var emb = GetWordEmbedding(tokens[t]);
            for (int d = 0; d < EmbeddingDim; d++)
                values[t * EmbeddingDim + d] = emb[d];
        }
        return new Tensor<double>(values, [1, MaxSeqLen, EmbeddingDim]);
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

    // ─── Decoding helpers ────────────────────────────────────────────────

    private static string[] Tokenize(string sentence) =>
        sentence.Split(' ', StringSplitOptions.RemoveEmptyEntries);

    private static string[] DecodeLabelTensor(Tensor<double> labelTensor, int tokenCount)
    {
        var labels = new string[tokenCount];
        for (int t = 0; t < tokenCount; t++)
        {
            // Label tensor shape: [1, seqLen] or [seqLen]
            int idx = labelTensor.Rank == 2
                ? (int)Math.Round(labelTensor[0, t])
                : (int)Math.Round(labelTensor[t]);

            idx = Math.Clamp(idx, 0, LabelNames.Length - 1);
            labels[t] = LabelNames[idx];
        }
        return labels;
    }

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

    // ─── System info ─────────────────────────────────────────────────────

    private static PocSystemInfo BuildSystemInfo() => new(
        Os: RuntimeInformation.OSDescription,
        Framework: RuntimeInformation.FrameworkDescription,
        LibraryVersion: typeof(AiModelBuilder<,,>).Assembly
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
    [property: JsonPropertyName("hiddenDim")] int HiddenDim,
    [property: JsonPropertyName("numLabels")] int NumLabels,
    [property: JsonPropertyName("useCRF")] bool UseCRF,
    [property: JsonPropertyName("trainingSentences")] int TrainingSentences,
    [property: JsonPropertyName("labelSet")] string[] LabelSet);

public sealed record PocTimings(
    [property: JsonPropertyName("totalMs")] double TotalMs,
    [property: JsonPropertyName("trainMs")] double TrainMs,
    [property: JsonPropertyName("inferenceMs")] double InferenceMs);

public sealed record PocSystemInfo(
    [property: JsonPropertyName("os")] string Os,
    [property: JsonPropertyName("framework")] string Framework,
    [property: JsonPropertyName("libraryVersion")] string LibraryVersion);
