# Classification POC — Ticket Compliance Analysis

**Artifacts reviewed:**
- `HttpResponses/response-classification-predict-text.json` (output of `POST /api/Classification/PredictText`, 4 labeled messages)
- `Controllers/ClassificationController.cs` (facade, feature extractor, controller)
- `ticket.md` (POC user story and acceptance criteria)

---

## Verdict: **PASS (with documented caveats)**

The ticket asks for a working, documented Classification POC that verifies whether the
AiDotNet library delivers on its claimed capabilities — not for a production-grade spam
filter. By that standard the POC passes every acceptance criterion. The one wrong
classification in the JSON (a phishing email predicted `not-spam`) is a **documented,
expected model limitation**, predicted in advance by the test suite
(`HttpRequests/classification.http`, test 15) — it is evidence the POC's evaluation is
honest, not evidence the implementation is broken. It must, however, be part of the
findings presented to the team, because it bounds what this model could be used for.

---

## 1. Ticket requirements vs. delivery

| Ticket requirement | Status | Evidence |
|---|---|---|
| POC created for Classification model type | ✅ Met | `FTTransformerNetwork<double>` (AiDotNet) trained and serving behind `ISpamClassificationFacade`; binary spam classification end-to-end |
| Includes setup | ✅ Met | One-time lazy training in `TransformerSpamClassificationFacade.TrainModel()` (100 samples, early-stop at loss ≤ 5e-4); singleton registration so requests pay inference only |
| Includes configuration | ✅ Met | Architecture fully specified in code and echoed in the response: `facadePattern` documents FeatureTokenizer(8→64) → CLS → 2× TransformerEncoder(4 heads) → Dense(2, softmax); `modelInfo` reports 100,802 parameters, 20 epochs, final loss 0.000363 |
| Working test scenario | ✅ Met | `HttpRequests/classification.http` — 17 requests covering easy/hard/edge/validation cases, each with verified expected outcomes; the JSON under review is one such scenario executed successfully |
| Results documented | ✅ Met | Expected outcomes are written next to every test with measured values; this summary plus the captured response complete the documentation |
| Sample input/output captured | ✅ Met | `response-classification-predict-text.json` IS the captured sample I/O: raw text in, extracted features + prediction + score out |
| Findings presented to team | ⚠️ Not verifiable from artifacts | Process step — the material exists (this document is presentation-ready), but presentation itself cannot be confirmed from the repo |
| Verify library delivers on claimed capabilities | ✅ Met (nuanced) | The POC proves the `FTTransformerNetwork.Train` path works (loss 0.75 → 0.0004, 100% train accuracy). It also documents where the library does NOT deliver: the entire `*Classifier.TrainStep` family is broken and `SaveModel`/`LoadModel` does not round-trip (code comments, `ClassificationController.cs` ~lines 55–72) — exactly the kind of verification the ticket asked for |

---

## 2. Expected vs. actual classification behavior

All four results in the JSON match the **pre-registered expectations** in
`classification.http` (tests 14–15), including the failure:

| # | Message | Label | Predicted | rawScore | Expected (from test suite) | Match |
|---|---|---|---|---|---|---|
| 1 | "CONGRATULATIONS WINNER!!! … $$$ … act now …" | spam | spam | 0.999774 | spam ≈ 0.9997 | ✅ |
| 2 | 85-word project status email | not-spam | not-spam | 0.000243 | not-spam ≈ 0.0003 | ✅ |
| 3 | "Thanks! See you at the meeting at 3pm." | not-spam | not-spam | 0.000479 | not-spam ≈ 0.0005 (out-of-distribution short legit — zero keyword counts win) | ✅ |
| 4 | Polished account-suspension phishing | spam | **not-spam** | 0.000511 | **predicted false negative** ≈ 0.0007 | ✅ (failure was predicted) |

`accuracy: 0.75` is arithmetically correct (3/4) and consistent with
`ComputeAccuracy` in the controller (all samples labeled → accuracy emitted).

**Feature extraction is correct.** Spot-check of message 1 against
`SpamFeatureExtractor.Extract`:
`[25, 0.4308, 8, 2, 6, 4, 0.08, 4.5909]` — 25 whitespace tokens; 8 `!`; 2 URLs
(`bit.ly`, `win-big.io`); 6 money hits (WINNER, FREE, cash, PRIZE, $$$, plus `win`
matched inside the `win-big.io` URL); 4 urgency hits (expires, act now, LIMITED,
HURRY); linkRatio 2/25 = 0.08; avgWordLen excludes URL tokens as designed. Message 2
correctly extracts all-zero keyword counts. The echoed features make the derivation
auditable, as intended.

---

## 3. Gaps, inconsistencies, and incorrect behaviors

### 3.1 Model-quality gap (significant — must be in the team findings)
The phishing false negative (row 4) is **confident**, not borderline: rawScore
0.000511 means the model is ~99.95% sure a phishing email is legitimate. Root cause is
the training data, not the code: the 100 synthetic training samples make the two
classes separable on surface statistics alone (spam = short/shouty/small words), so
the trained network keys on `wordCount`/`capsRatio`/`avgWordLen` and effectively
ignores `moneyKeywords`/`urgencyKeywords`/`urlCount` — even though `moneyKeywords ≥ 2`
is a perfect class separator in the training set. Any professionally written spam
sails through. The model is also uncalibrated: every score in the JSON is saturated
(< 0.001 or > 0.999); there is no "unsure" zone.

### 3.2 Extractor keyword coverage (contributing factor)
The phishing message extracted only `moneyKeywords=1, urgencyKeywords=1` because the
classic phishing vocabulary it uses — *refund, charge, billing, verify, suspension,
"within 24 hours"* — is absent from `SpamFeatureExtractor`'s pattern lists (which
cover free/$$$/winner/prize/% off/cash/win and act now/urgent/expires/limited/hurry/
last chance). Even a keyword-sensitive model would have seen weak signals here.

### 3.3 Non-determinism despite a declared seed (minor)
`facadePattern` advertises `Seed=42`, but this run reports `trainEpochs: 20,
finalTrainLoss: 0.000363` while an earlier run of the same build reported 19 epochs /
0.00039. The seed is applied to `FTTransformerOptions` and `PrependCLSTokenLayer`, but
other layer initializations (e.g. `TransformerEncoderLayer`, `DenseLayer`) are not
seeded, so training is only approximately reproducible across processes. Harmless for
a POC; misleading if anyone relies on the seed for exact reproduction.

### 3.4 Reporting quirks (cosmetic, worth a footnote)
- `timings.totalMs: 14.378` is *smaller* than `timings.trainMs: 4323.474`. Not a bug:
  `trainMs` is the cached one-time training cost from process start, while `totalMs`
  covers only this request (the model was already trained). The field naming invites
  misreading.
- `system.libraryVersion: "0.204.0+…"` while the installed package is 0.213.3 — a
  known stale assembly attribute in the AiDotNet package, already flagged by a code
  comment (`// NB: AiDotNet 0.213.3 ships with stale assembly version attributes`).

### 3.5 Ticket-side gap
"Findings are presented to the team for review" is a process step that no repo
artifact can demonstrate. The presentation material exists; the presentation is the
remaining action.

---

## 4. Recommended improvements (if the POC graduates beyond proof-of-concept)

1. **Fix the false-negative class, in the data.** Add a "polished phishing" cluster to
   the synthetic training set — legit-like surface statistics (wordCount 60–150,
   capsRatio < 0.1, exclamations 0–1, avgWordLen 5–6) combined with spam-like signals
   (urlCount 2–4, moneyKeywords 1–3, urgencyKeywords 1–3) labeled spam. That forces
   the network to weight the keyword features instead of prose statistics. A threshold
   change cannot fix this (the score is 0.0005, not 0.4).
2. **Broaden the extractor's phishing vocabulary**: add refund/charge/billing/account/
   claim to the money list and verify/suspension/"within \d+ hours"/immediately to the
   urgency list, then retrain against data generated under the same semantics.
3. **Surface uncertainty honestly**: scores are saturated everywhere; if downstream
   consumers will see `rawScore`, either calibrate (e.g. temperature scaling against a
   held-out set) or document that the score is a decision, not a probability.
4. **Seed all layers** (or document approximate reproducibility) so the advertised
   `Seed=42` matches observed behavior across runs.
5. **Rename or annotate timing fields** (`trainMs` → `oneTimeTrainMs` or add a
   `modelTrainedThisRequest: bool`) so `totalMs < trainMs` cannot be misread.
6. **Close the loop on the ticket**: present these findings — including the verified
   library defects (broken `*Classifier.TrainStep` family, broken save/load
   round-trip) and the phishing blind spot — as the team-review deliverable.

---

## Bottom line

The implementation does exactly what it claims, the captured output matches the
pre-registered expectations case for case (including the predicted failure), and every
verifiable acceptance criterion in `ticket.md` is satisfied → **PASS as a POC**. The
same evidence shows the *model* is not deployable for real spam filtering (confident
false negatives on polished phishing, uncalibrated scores, surface-statistics bias) —
which is precisely the kind of finding the ticket was written to surface.
