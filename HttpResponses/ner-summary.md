# NER POC — Ticket Compliance Analysis

**Artifacts reviewed:**
- `HttpResponses/response-ner-predict-text.json` (output of `POST /api/Ner/Analyze`, 21 sentences — the full scenario set from `HttpRequests/ner.http`)
- `Controllers/NerController.cs` (facade, transformer tagger, BIO span extraction)
- `ticket.md` (POC user story and acceptance criteria)

---

## Verdict: **PASS (as a POC), with one behavior every reviewer must understand**

The ticket asks for a working, documented NER POC that verifies AiDotNet's claimed
capabilities. That is delivered: the model trains (58 epochs to loss 0.009972),
serves, and reproduces its training labels perfectly; the test suite pre-registered
expected outcomes for every scenario in this JSON, and the output matches them —
including the failures, which were predicted in advance. The critical caveat: **this
model is a word-memorization table, not a generalizing NER system**, and on unseen
words it hallucinates entities nondeterministically (different ones per server
restart). Both behaviors are documented in `ner.http`; both must be in the findings
presented to the team.

Note: this response comes from a different process than the run documented in
`ner.http` (58 training epochs here vs ~65 there). That matters because the
controller's hash-based embeddings use `string.GetHashCode()`, which .NET randomizes
per process — so out-of-vocabulary results in this JSON legitimately differ in their
*specifics* from the suite's "(this run)" examples while matching their documented
*character*. In-vocabulary results are identical across both runs, exactly as the
suite's header predicts.

---

## 1. Ticket requirements vs. delivery

| Ticket requirement | Status | Evidence |
|---|---|---|
| POC created for NER model type | ✅ Met | Per-token transformer tagger (`NeuralNetwork<double>`, stock AiDotNet layers: Dense(50→64) → 2× TransformerEncoder → Dense(9, softmax)) behind `INerFacade`; full BIO tagging + span extraction pipeline |
| Includes setup | ✅ Met | One-time lazy training in `TransformerNerFacade.TrainModel()` on 5 CoNLL-style sentences, early-stop at loss ≤ 0.01; singleton so requests pay inference only (`inferenceMs: 51` for 21 sentences, batched as one `[21,15,50]` tensor) |
| Includes configuration | ✅ Met | `modelInfo` reports the full configuration: 9 BIO labels, maxSeqLen 15, 103,433 parameters, 58 epochs, final loss 0.009972; `facadePattern` documents the layer stack and training contract |
| Working test scenario | ✅ Met | `HttpRequests/ner.http` — 11 requests with verified expected outcomes; this JSON is the executed scenario set (memorization, recombination, OOV, case-blindness, punctuation, truncation, orphan-I, batch) |
| Results documented | ✅ Met | Every scenario's expected outcome is documented next to the request, including which behaviors are stable and which are per-process noise |
| Sample input/output captured | ✅ Met | This JSON is the captured sample I/O: sentence in → tokens, per-token BIO labels, decoded entity spans out |
| Findings presented to team | ⚠️ Not verifiable from artifacts | Process step; this document is the presentation-ready material |
| Verify library delivers on claimed capabilities | ✅ Met (nuanced) | Proves the stock-layer `NeuralNetworkBase.Train` path works for sequence tagging. Also documents what does NOT work in AiDotNet 0.213.3: `BiLSTMCRF.Train` diverges and `AiModelBuilder.BuildAsync` breaks on tiny datasets (code comments, `NerController.cs` ~lines 53–64) — the verification the ticket asked for |

---

## 2. Expected vs. actual NER behavior

### 2.1 In-vocabulary behavior — 100% as expected (stable across runs)

- **Training sentences (results 1–5)**: perfect label reproduction. e.g. result 1:
  `Apple Inc → ORG`, `San Francisco → LOC` with correct token spans; result 4 keeps
  `European = O`, faithfully reproducing the training data's own quirk (CoNLL would
  tag it B-MISC — the model learned the data it was given, including its flaws).
- **Recombined vocabulary (results 6–8, 16–17)**: novel sentences built from trained
  words tag perfectly — "Tesla acquired SpaceX in London" → `[Tesla=ORG] [SpaceX=ORG]
  [London=LOC]`; "Barack Obama leads Google in Paris" → all three entities correct.
  As the suite warns: labels follow the WORD, not the context. Correct-looking
  output here is memorization, not understanding.
- **Case-blindness (results 10, 20)**: "i ate an apple … in paris with elon" →
  `[apple=ORG] [paris=LOC] [elon=PER]` — confident false positives on common-noun
  usage, exactly as predicted (embeddings lowercase before hashing).
- **Orphan I- tags (result 13)**: "Musk and Obama met in Francisco" → bioLabels
  `I-PER, O, I-PER, O, O, I-LOC` and **`entities: []`**. Three recognizable entities
  visible in the labels, zero reported — `ExtractEntitySpans` (NerController.cs
  ~line 319) only opens spans on `B-`, silently discarding orphan `I-` tags instead
  of repairing them.
- **Punctuation glue (results 11, 21)**: `"Paris,"` and `"Tesla,"`/`"Google,"` → `O`
  — one comma costs the model entities it provably knows (space-only `Tokenize`,
  ~line 307).
- **Truncation (result 12)**: 18-token sentence — `San`(16) and `Francisco`(17) are
  beyond `MaxSeqLen=15` and forced to `O` by `DecodeBioLabels` (~line 191) without
  the model ever seeing them. Entity silently lost; no warning in the response.

### 2.2 Out-of-vocabulary behavior — hallucination, randomized per process (as documented)

The suite's header warns that OOV tags are per-process noise; this JSON confirms it
by differing from the suite's recorded examples while matching their character:

| Result | Sentence | This run's hallucinations | Suite's documented run |
|---|---|---|---|
| 9 | "Microsoft was founded by Bill Gates in Seattle" | `[founded=PER]`; Microsoft/Bill/Gates/Seattle all `O` | `[Bill=ORG]`, Microsoft=I-LOC |
| 12 | truncation filler | `[large=ORG]`, `[on=LOC]` | `[build=MISC]`, `[large=ORG]` |
| 18 | "Amazon opened new offices in Toronto…" | all `O`, no entities | `[Amazon=ORG]` (lucky), `[opened=LOC]` |
| 19 | "Taylor Swift performed at Wembley Stadium…" | `[Wembley=ORG] [Stadium=ORG] [on=LOC]`; Taylor Swift missed | `[Wembley=ORG]`; Taylor Swift missed |
| 20 | "my friend elon moved to paris…" | `[my=ORG]`, `[to=LOC]` (plus correct elon/paris) | `spring=I-PER` stray |
| 11 | "Visit Paris, London and Tesla." | `"Tesla."` → `O` (entity lost) | `"Tesla."` → B-ORG (lucky hallucination) |

Two takeaways, both pre-registered in `ner.http`: (1) the model **hallucinates
rather than abstains** on unseen words — function words like *on*, *my*, *to*,
*this* are reported as entities; (2) the same request can return **different
entities after a server restart**, because embeddings derive from the
process-randomized `GetHashCode()` (`GetWordEmbedding`, ~line 299).

### 2.3 Response-shape consistency with the code

- `entities` spans are consistent with `bioLabels` under the B-opens/I-extends rule
  everywhere, including multi-token spans (`startToken`/`endToken` correct).
- `modelInfo` matches the facade constants (9 labels, maxSeqLen 15, 5 training
  sentences); `trainEpochs: 58` is within the documented ~60-epoch convergence.
- `timings.totalMs: 51` < `trainMs: 4892` — same naming quirk as the classification
  POC: `trainMs` is the cached one-time training cost, this request paid inference
  only.
- `system.libraryVersion: "0.204.0+…"` — known stale assembly attribute in the
  AiDotNet 0.213.3 package (flagged by code comment).

---

## 3. Gaps, inconsistencies, and incorrect behaviors

1. **No generalization — by construction** (significant). Hash-based embeddings carry
   zero semantics, and 5 sentences cover ~40 words. Result 18 ("Amazon opened new
   offices in Toronto") finds nothing; result 19 misses Taylor Swift entirely. Fine
   for a library POC; disqualifying for any real NER use.
2. **Hallucinated entities on OOV input** (significant). `[founded=PER]` (result 9),
   `[on=LOC]` (results 12, 19), `[my=ORG]`, `[to=LOC]` (result 20), `[this=ORG]`
   (result 21). The model has no abstention mechanism and the response carries no
   confidence scores, so consumers cannot filter the noise.
3. **Nondeterministic output across restarts** (significant, surprising). Identical
   input produces different entities after a process restart — `string.GetHashCode()`
   is randomized per process. The most counterintuitive behavior in the POC; must be
   communicated to the team.
4. **Orphan I- spans silently dropped** (moderate, decoder bug-by-design). Result 13
   shows three known entities in `bioLabels` and an empty `entities` list. Standard
   CoNLL decoding repairs `I-X` starts to `B-X`.
5. **Punctuation destroys recall** (moderate). "Paris," ≠ "Paris" under the
   space-only tokenizer; results 11 and 21 lose 3 known entities to commas/periods.
6. **Silent truncation at 15 tokens** (moderate). Result 12 drops `San Francisco`
   with no indication in the response that the sentence exceeded the model window.
7. **Cosmetic**: `trainMs` vs `totalMs` naming; stale `libraryVersion`.

None of these contradict the ticket — they are exactly the "findings" a POC exists to
produce, and all are pre-documented with verified expectations in `ner.http`.

---

## 4. Recommended improvements (if the POC graduates)

1. **Replace hash embeddings with real ones** (pretrained GloVe/fastText vectors, or
   at minimum a *stable* hash such as SHA-256-derived) — fixes both the zero-semantics
   generalization wall and the per-restart nondeterminism in one move.
2. **Repair BIO sequences in `ExtractEntitySpans`**: treat a leading `I-X` as `B-X`
   (one-line change) so result-13-style detections aren't silently discarded.
3. **Tokenize properly**: strip punctuation or use a regex word tokenizer so
   "Paris," matches the trained "Paris".
4. **Surface confidence + truncation**: include per-token max-softmax probability and
   a `truncated: true` flag when `tokens.Length > MaxSeqLen`, so consumers can filter
   hallucinations and detect lost tails.
5. **Add an abstention threshold**: report `O` unless the winning label's probability
   clears a floor — converts hallucinations into abstentions on OOV input.
6. **Scale the training data** before drawing any conclusions about tagging quality;
   5 sentences can only ever demonstrate the training pipeline, which is all the
   ticket required.

---

## Bottom line

Implementation and output are mutually consistent, every scenario in the JSON matches
the pre-registered expectations in `ner.http` (in-vocabulary exactly; out-of-vocabulary
in documented character, with the documented per-process randomness), and all
verifiable ticket criteria are satisfied → **PASS as a POC**. The same evidence shows
the model itself is a memorization demo — no generalization, nondeterministic
hallucinations on unseen words, and a span decoder that drops malformed-but-real
detections — which is precisely the verification of library capability the ticket was
written to obtain.
