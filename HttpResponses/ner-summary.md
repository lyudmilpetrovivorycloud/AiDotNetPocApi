# NER POC — Ticket Compliance Analysis

**Artifacts reviewed:**
- `HttpResponses/response-ner-predict-text.json` — output of `POST /api/Ner/Analyze`, 21 sentences (the executed scenario set from `HttpRequests/ner.http`)
- `Controllers/NerController.cs` — facade, per-token transformer tagger, BIO span extraction
- `ticket.md` — POC user story / acceptance criteria

> **Run note:** the examples below are quoted from *this* JSON. Out-of-vocabulary (OOV)
> tags differ from those documented in `ner.http` because the controller derives word
> embeddings from `string.GetHashCode()`, which .NET randomizes **per process**
> (`GetWordEmbedding`, NerController.cs:309-317). In-vocabulary results are identical
> across runs; OOV specifics are per-process noise. `modelInfo` here: `trainEpochs: 59`,
> `finalTrainLoss: 0.009955`, 103,433 parameters.

---

## Verdict: **PASS (as a POC), with one behavior every reviewer must understand**

The ticket asks for a working, documented NER POC that verifies AiDotNet delivers on its
claimed capabilities. That is delivered: the model trains (59 epochs to loss 0.009955),
serves 21 sentences in one batched inference pass, and reproduces its training labels
perfectly. The ticket sets **no accuracy threshold**, so the model's limitations do not
fail it.

The critical caveat for the team: **this model is a word-memorization table, not a
generalizing NER system.** On unseen words it hallucinates entities, and the specific
hallucinations change on every server restart. Both behaviors are correct/expected per
the implementation and must headline the findings presentation.

---

## 1. Ticket requirements vs. delivery

| Ticket requirement | Status | Evidence |
|---|---|---|
| POC created for NER | ✅ | Per-token transformer tagger (`NeuralNetwork<double>`, stock layers Dense(50→64) → 2× TransformerEncoder(4 heads) → Dense(9, softmax)) behind `INerFacade`; full BIO + span-extraction pipeline |
| Setup | ✅ | One-time lazy training in `TrainModel()` on 5 CoNLL-style sentences, early-stop at loss ≤ 0.01; singleton so requests pay inference only |
| Configuration | ✅ | `modelInfo` reports 9 BIO labels, maxSeqLen 15, 103,433 params, 59 epochs, loss 0.009955; `facadePattern` documents the full layer/training contract |
| Working test scenario | ✅ | `HttpRequests/ner.http` (11 requests, pre-registered expectations); this JSON is the executed result set |
| Results documented | ✅ | Response carries `results`, `modelInfo`, `facadePattern`, `interfaceChain`, `timings`, `system` |
| Sample input/output captured | ✅ | This JSON: sentence → tokens → per-token `bioLabels` → decoded `entities` |
| Findings presented to team | ⚠️ Process step | This document is the presentation-ready material |
| Verify library capability | ✅ (nuanced) | Proves the stock-layer `NeuralNetworkBase.Train` path works for sequence tagging; code comments (NerController.cs:65-76) also document what does **not** work in 0.213.3 — `BiLSTMCRF.Train` diverges, `AiModelBuilder.BuildAsync` breaks on tiny datasets |

---

## 2. Expected vs. actual NER behavior

### 2.1 In-vocabulary — 100% as expected (stable across runs)

- **Exact training sentences (results 1-5):** perfect label reproduction. Result 1
  `Apple Inc → ORG`, `San Francisco → LOC`; result 4 keeps `European = O`, faithfully
  reproducing the training data's own quirk (CoNLL would tag it MISC — the model learned
  the data it was given).
- **Recombined vocabulary (results 6-8, 14-17):** novel sentences built from trained
  words tag perfectly — result 6 "Tesla acquired SpaceX in London" →
  `[Tesla=ORG] [SpaceX=ORG] [London=LOC]`; result 17 "Barack Obama leads Google in Paris"
  → all three correct. The stack has no positional encoding, so **labels follow the word,
  not the context** — correct-looking output here is memorization, not understanding.
- **Case-blindness (results 10, 20):** result 20 "my friend elon moved to paris last
  spring" → `[elon=PER] [paris=LOC]`; result 10 "i ate an apple … in paris with elon" →
  `[apple=ORG] [paris=LOC] [elon=PER]` — confident false positive on the common-noun
  "apple" because embeddings lowercase before hashing.

### 2.2 Structural decoder limits — confirmed in this run (stable)

- **Orphan I- tags dropped (result 13):** "Musk and Obama met in Francisco" →
  `bioLabels: [I-PER, O, I-PER, O, O, I-LOC]` but **`entities: []`**. Musk/Obama/Francisco
  were trained only as I- continuations (of Elon/Barack/San), so standalone they emit I-
  tags; `ExtractEntitySpans` (NerController.cs:324-354) only opens a span on `B-` and
  silently discards every orphan `I-`. Three recognizable entities in the labels, zero
  reported.
- **Punctuation glue (results 11, 21):** the space-only `Tokenize` (NerController.cs:319)
  leaves punctuation attached, creating OOV tokens. Result 11 "Visit Paris, London and
  Tesla." → `"Paris,"`→O and `"Tesla."`→O (both known entities **lost** to one comma/period),
  only `London=LOC` survives. Result 21 "Tesla, SpaceX and Google, all grew this year" →
  `"Tesla,"`→O and `"Google,"`→O lost, only bare `SpaceX=ORG` survives.
- **Silent truncation at 15 tokens (result 12):** "Tesla plans to build … near beautiful
  San Francisco" is 18 tokens; `San`(16) and `Francisco`(17) sit beyond `MaxSeqLen=15` and
  are forced to `O` by `DecodeBioLabels` (NerController.cs:203-207) without the network
  ever seeing them. `San Francisco` is silently lost — `bioLabels` still has 18 entries and
  the response carries no truncation flag.

### 2.3 Out-of-vocabulary — hallucination, not abstention (per-process noise)

This run's specific OOV outcomes (differ from `ner.http`'s documented run, same character):

| Result | Sentence | This run's OOV behavior |
|---|---|---|
| 9 | "Microsoft was founded by Bill Gates in Seattle" | **all `O`, `entities: []`** — this run happened to abstain; no entity recognized (Microsoft/Bill Gates/Seattle all missed) |
| 10 | "…apple…paris with elon" | `[with=LOC]` hallucinated alongside the correct in-vocab tags |
| 11 | "Visit Paris, London and Tesla." | `[Visit=PER]` hallucinated |
| 12 | "Tesla plans to build…" | `[somewhere=ORG]` hallucinated; `build=I-ORG` orphan (dropped) |
| 18 | "Amazon opened new offices in Toronto last quarter" | `[Amazon=PER]` (wrong type), `[opened=PER]` hallucinated; Toronto missed |
| 19 | "Taylor Swift performed at Wembley Stadium on Friday" | `[Stadium=LOC] [Friday=LOC]` hallucinated; Taylor Swift missed; `Wembley=I-LOC` orphan (dropped) |
| 21 | "Tesla, SpaceX and Google, all grew this year" | `[grew=ORG]` hallucinated |

Two takeaways, both pre-registered in `ner.http`: (1) the model **hallucinates rather than
abstains** on unseen words — function/common words (`with`, `Visit`, `somewhere`, `opened`,
`grew`) are reported as entities; (2) identical input can return **different entities after a
restart** (note result 9 abstained this run but hallucinated `[Bill=ORG]` in the `ner.http`
run), because embeddings derive from process-randomized `GetHashCode()`.

### 2.4 Response-shape consistency

- `entities` are consistent with `bioLabels` under the B-opens/I-extends rule everywhere,
  including multi-token spans (`startToken`/`endToken` correct, e.g. `Apple Inc` 0-1).
- `modelInfo` matches facade constants (9 labels, maxSeqLen 15, 5 training sentences);
  `trainEpochs: 59` within the documented ~60-epoch convergence.
- `timings.totalMs: 52.376` ≈ `inferenceMs: 52.351` ≪ `trainMs: 6420.459` — expected:
  training is the cached one-time cost, this request paid inference only.
- `system.libraryVersion: "0.204.0+…"` — known stale assembly attribute in AiDotNet 0.213.3
  (flagged by code comment, NerController.cs:421-422).

---

## 3. Gaps, inconsistencies, and incorrect behaviors

1. **No generalization — by construction** (significant). Hash-based embeddings carry zero
   semantics and 5 sentences cover ~30 unique tokens. Result 9 finds nothing; result 19
   misses Taylor Swift entirely. Acceptable for a library POC; disqualifying for real use.
2. **Hallucinated entities on OOV input** (significant). `[with=LOC]`, `[Visit=PER]`,
   `[somewhere=ORG]`, `[opened=PER]`, `[grew=ORG]`. No abstention mechanism and no
   confidence scores in the response, so consumers cannot filter the noise.
3. **Nondeterministic output across restarts** (significant, surprising). Same input,
   different entities after a process restart — `string.GetHashCode()` is per-process
   randomized. The most counterintuitive behavior; must be communicated to the team.
4. **Orphan I- spans silently dropped** (moderate, decoder bug-by-design). Result 13: three
   known entities in `bioLabels`, empty `entities`. A real CoNLL decoder repairs leading
   `I-X` → `B-X`.
5. **Punctuation destroys recall** (moderate). Results 11 & 21 lose 3 known entities
   (`Paris,`, `Tesla,`, `Google,`) to attached commas/periods.
6. **Silent truncation at 15 tokens** (moderate). Result 12 drops `San Francisco` with no
   indication in the response.
7. **Cosmetic**: stale `libraryVersion`; `trainMs` vs `totalMs` naming.

None of these contradict the ticket — they are exactly the findings a POC exists to produce.

---

## 4. Recommended improvements (if the POC graduates)

1. **Replace hash embeddings with real ones** (pretrained GloVe/fastText, or at minimum a
   *stable* hash such as SHA-256-derived) — fixes both the zero-semantics generalization wall
   and the per-restart nondeterminism in one move.
2. **Repair BIO sequences in `ExtractEntitySpans`**: treat a leading `I-X` as `B-X` so
   result-13-style detections aren't silently discarded.
3. **Tokenize properly**: strip/split punctuation (regex word tokenizer) so "Paris," matches
   trained "Paris".
4. **Surface confidence + truncation**: include per-token max-softmax probability and a
   `truncated: true` flag when `tokens.Length > MaxSeqLen`.
5. **Add an abstention threshold**: emit `O` unless the winning label clears a probability
   floor — converts OOV hallucinations into abstentions.
6. **Scale the training data** before drawing any conclusion about tagging quality; 5
   sentences can only demonstrate the training pipeline, which is all the ticket required.

---

## Bottom line

Implementation and output are mutually consistent, every scenario in the JSON matches the
pre-registered expectations in `ner.http` (in-vocabulary exactly; out-of-vocabulary in
documented *character*, with the documented per-process randomness), and all verifiable
ticket criteria are satisfied → **PASS as a POC**. The same evidence shows the model is a
memorization demo — no generalization, nondeterministic hallucinations on unseen words, a
span decoder that drops malformed-but-real detections, and recall lost to punctuation and a
15-token window — which is precisely the library-capability verification the ticket was
written to obtain.
