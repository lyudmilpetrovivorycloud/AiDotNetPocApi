# Classification POC — Evaluation Summary

**Endpoint analyzed:** `POST /api/Classification/PredictText`
**Artifacts:** `response-classification-predict-text.json`, `Controllers/ClassificationController.cs`, `ticket.md`
**Model:** `FTTransformerNetwork<double>` (FT-Transformer, custom CLS-readout layer stack), AiDotNet 0.213.3

---

## Conclusion: **PASS (with a documented model-quality caveat)**

Against the **ticket's stated requirements**, the classification POC **passes**. The ticket asks for
a working classification POC — setup, configuration, a working test scenario, documented results,
and captured sample input/output — and **does not define any accuracy threshold or per-message
correctness guarantee**. All of those deliverables are present and the endpoint runs end-to-end on
real text.

Against a stricter reading ("does the classifier actually classify correctly"), the result is a
**partial pass: 75% accuracy (3/4), with one false negative on a realistic phishing email.** This
is a model/training-data limitation, **not a code defect**.

---

## Ticket requirements vs. implementation

| Ticket requirement | Status | Evidence |
|---|---|---|
| Classification POC exists | ✅ | `ClassificationController` with `Predict` (raw features) and `PredictText` (raw text) endpoints |
| Setup & configuration | ✅ | Facade-encapsulated model, trained once per process; arch + hyperparams in `BuildNetwork()` / `FTTransformerOptions` |
| Working test scenario | ✅ | JSON response classifies 4 messages end-to-end with timings |
| Results documented | ✅ | Response includes `modelInfo`, `accuracy`, `timings`, `facadePattern`, `interfaceChain`, `system` |
| Sample input/output captured | ✅ | `response-classification-predict-text.json` echoes text, extracted features, score, prediction, correctness |
| Findings presented to team | ⚠️ | This document; flag the phishing false negative below |

The ticket sets **no accuracy bar**, so the 75% figure does not by itself fail the ticket — but it
must be surfaced to the team as a capability finding.

---

## Expected vs. actual classification behavior

| # | Message (abridged) | Label (expected) | Prediction | Score P(spam) | Correct |
|---|---|---|---|---|---|
| 1 | "CONGRATULATIONS WINNER!!! … FREE cash PRIZE … HURRY!!" | 1 (spam) | spam | 0.9997 | ✅ |
| 2 | "Hi team, following up on yesterday's planning meeting…" | 0 (not-spam) | not-spam | 0.0003 | ✅ |
| 3 | "Thanks! See you at the meeting at 3pm." | 0 (not-spam) | not-spam | 0.0009 | ✅ |
| 4 | "Dear customer, we detected unusual activity… verify your billing… expires today…" | **1 (spam)** | **not-spam** | **0.0012** | ❌ |

**Accuracy reported: `0.75`** (matches `ComputeAccuracy`, 3/4).

> Note: scores drift slightly between runs (Msg 1 = 0.9997, Msg 4 = 0.0012 in the latest run vs.
> ~0.9998 / 0.0005 in an earlier captured run) — training is per-process and not persisted
> (`SaveModel` does not round-trip this stack). The verdicts and the 0.75 accuracy are stable.

---

## Gaps & root-cause analysis

### 1. False negative on polished phishing (Msg 4) — primary finding
Extracted features for Msg 4: `[66, 0.0238, 0, 2, 1, 1, 0.0303, 5.0625]`
(`wordCount, capsRatio, exclamations, urlCount, moneyKeywords, urgencyKeywords, linkRatio, avgWordLen`).

Compare to the synthetic training profiles in `GenerateTrainingData` (lines 300-323):

| Feature | Msg 4 | Spam profile (train) | Legit profile (train) | Falls in |
|---|---|---|---|---|
| capsRatio | 0.0238 | 0.3–0.7 | 0–0.1 | **legit** |
| exclamations | 0 | 3–10 | 0–2 | **legit** |
| linkRatio | 0.0303 | 0.2–0.5 | 0–0.05 | **legit** |
| avgWordLen | 5.06 | 3–5 | 5–7.5 | **legit** |
| urlCount | 2 | 2–6 | 0–2 | overlap |
| moneyKeywords | 1 | 2–5 | 0 | between |
| urgencyKeywords | 1 | 2–5 | 0–1 | legit edge |

**Root cause:** the training data encodes "spam = *loud*" (heavy caps, many exclamations, dense
links, money/urgency keyword spam). A grammatically polished phishing email exhibits none of those
surface signals, so nearly every feature lands inside the legit range and the model — which leans on
`capsRatio`/`exclamations` — confidently scores it legit (~0.001). This matches the recorded
model-behavior note: *surface stats dominate, keywords barely register, polished phishing = false
negative.*

### 2. Scores are saturated / poorly calibrated
All four scores sit at the extremes (~0.999 or ~0.0003–0.0012). The model converged to near-zero
loss (`finalTrainLoss` ≈ 0.0004) on **only 100 synthetic, linearly-separable samples** — it has
memorized the synthetic boundary rather than learned a calibrated decision surface. There is no
useful "uncertain" middle band, so a borderline real-world message gets a falsely confident verdict
rather than a low-confidence one a human could triage.

### 3. Feature extractor cannot recover the signals that matter
`SpamFeatureExtractor` only measures surface statistics. Real phishing relies on *semantics*
("verify your billing information", "avoid suspension", lookalike domains such as
`secure-verify.example.com`) that the 8-feature vector discards. The classifier can only be as good
as these features, and they miss the most dangerous spam class.

### 4. Minor / informational (not failures)
- `system.libraryVersion` reports `0.204.0` while the actual package is 0.213.3 — already explained
  by the in-code comment (stale assembly attribute shipped by the library); cosmetic only.
- `timings.totalMs` (≈19 ms) ≈ `inferenceMs` (≈19 ms) while `trainMs` ≈ 6180 ms — expected: the
  singleton trains lazily once, so this request paid inference cost only. Correct by design.
- `trainEpochs` 20 (early-stopped at loss ≤ 5e-4 before the 30-epoch cap) — consistent with code.

---

## Recommendations

These improve *model/POC quality*; they are not blockers for ticket sign-off:

1. **Flag the limitation to the team explicitly.** The honest POC finding is: "FT-Transformer wiring
   works and trains, but on this synthetic feature set it detects loud spam and misses polished
   phishing (75% on the 4-message probe)." That is itself a valid POC outcome under the ticket.
2. **Strengthen training data.** Add a "stealth/phishing" spam class to `GenerateTrainingData`: low
   caps, zero exclamations, professional tone, but malicious URLs and account/billing/verify
   language — so the model learns spam is not only *loud* spam.
3. **Add semantic features** to `SpamFeatureExtractor`: suspicious/lookalike domain detection,
   credential/billing-verification phrase patterns, urgency phrasing independent of `!`. Consider
   expanding beyond 8 hand-crafted features or using text embeddings.
4. **Calibrate / report confidence.** Train on more, noisier data so scores aren't saturated;
   surface a confidence band so borderline messages are flagged for review rather than auto-decided.
5. **Expand the evaluation probe.** Four messages is too small to characterize the model; use a
   held-out labeled set with precision/recall — especially **recall on spam**, the metric that
   matters for phishing — rather than a single accuracy number.

---

## Bottom line
The classification POC **satisfies the ticket's requirements** (working scenario, documented
results, captured sample I/O). The evaluation also surfaces a real capability gap — the model
misclassifies polished phishing as legitimate (Msg 4) — which should be reported as the headline
finding when presenting to the team, since it directly affects whether AiDotNet's classification is
adoption-ready for spam/phishing use cases.
