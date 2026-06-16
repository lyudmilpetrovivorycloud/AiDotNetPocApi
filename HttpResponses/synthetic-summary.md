# Synthetic Data POC — Ticket Compliance Analysis

**Artifacts reviewed:**
- `HttpResponses/response-synthetic-predict-text.json` — output of `POST /api/Synthetic/Generate`, 200 rows
- `Controllers/SyntheticController.cs` — facade, SMOTE-NC generator, stats/correlation reporting
- `ticket.md` — POC user story / acceptance criteria

> **Run note:** the numbers below are quoted from *this* JSON. The generator is seeded
> (Seed=42) but `Generate` advances a shared RNG across requests (`synthetic.http` test 6),
> so a given draw differs run-to-run while the structural guarantees stay invariant. This
> artifact: `generatedSamples: 200`, `timings.totalMs: 9.859`, `trainMs: 13.704`,
> `inferenceMs: 9.299`. The `syntheticStats` min/max are computed by `ComputeStats` directly
> from the 200 rows, so they are the true row extremes — the hull check below relies on that.

---

## Verdict: **PASS** — the cleanest of the three POCs

The ticket asks for a working, documented synthetic-generation POC that verifies AiDotNet's
claimed capabilities. This artifact satisfies every verifiable acceptance criterion, and —
unlike the Classification and NER POCs — there is **no behavioral asterisk**: every structural
guarantee the implementation promises (convex hull, exact categoricals, correlation
preservation) holds on all 200 generated rows. The caveats are interpretive (what SMOTE
fidelity does and does not *mean*), not defects. The ticket sets no numeric threshold; this
POC would clear a demanding one anyway.

---

## 1. Ticket requirements vs. delivery

| Ticket requirement | Status | Evidence |
|---|---|---|
| POC created for Synthetic generation | ✅ | `SMOTENCGenerator<double>` via the library's first-class `ISyntheticTabularGenerator` API (`Fit(Matrix, ColumnMetadata, epochs)` → `Generate(n)`) behind `ISyntheticDataFacade` |
| Setup | ✅ | One-time lazy fit on 50 hardcoded customer rows (`FitGenerator()`, SyntheticController.cs:147-160); singleton, fit cost cached for all requests |
| Configuration | ✅ | `modelInfo` echoes K=5, 50 seed rows, generatedSamples, and the full column schema with **Education declared Categorical** (HighSchool/Bachelor/Master/PhD) — the schema is actually passed to `Fit`, the "NC" in SMOTE-NC |
| Working test scenario | ✅ | `HttpRequests/synthetic.http` — 10 requests (fidelity audit, boundary+privacy audit, degenerate n=1, RNG behavior, clamping, validation) with pre-registered expectations; this JSON is the max-scale 200-row scenario |
| Results documented | ✅ | Response carries its own evaluation: `realStats` vs `syntheticStats` vs `correlations` |
| Sample input/output captured | ✅ | This JSON — 200 generated rows plus the full fidelity report |
| Findings presented to team | ⚠️ Process step | This document is the presentation-ready material |
| Verify library capability | ✅ (nuanced) | SMOTENC delivers; code comments (SyntheticController.cs:56-72) document what does **not** on 0.213.3 — CTGAN diverges on small data (worse with more epochs), Copula degrades correlations to 0.82–0.93 and leaks float noise into categoricals — the comparative verification the ticket asked for |

---

## 2. Expected vs. actual behavior — every guarantee verified on this artifact

### 2.1 Statistical fidelity (the headline metric) — excellent

| Column | Real mean / std | Synthetic mean / std | Assessment |
|---|---|---|---|
| Age | 38.2 / 11.056 | 38.631 / 10.779 | mean +1.1%, std slightly below real |
| Income | 72,500 / 31,665.281 | 73,607.39 / 30,738.166 | mean +1.5%, std slightly below real |
| Education | 1.68 / 1.067 | 1.74 / 0.996 | close; std slightly below |
| CreditScore | 706.84 / 82.138 | 711.45 / 78.411 | mean +0.65%, std slightly below |

All six pairwise correlations preserved within **0.01** of real:

| Pair | Real | Synthetic | Δ |
|---|---|---|---|
| Age~Income | 0.9893 | 0.9902 | 0.0009 |
| Age~Education | 0.9484 | 0.9410 | 0.0074 |
| Age~CreditScore | 0.9535 | 0.9515 | 0.0020 |
| Income~Education | 0.9528 | 0.9508 | 0.0020 |
| Income~CreditScore | 0.9681 | 0.9661 | 0.0020 |
| Education~CreditScore | 0.9681 | 0.9649 | 0.0032 |

This correlation preservation is the property that justified choosing SMOTENC over CTGAN
(diverged) and Copula (correlations degraded to 0.82–0.93) on this seed set, and the artifact
confirms it at the maximum allowed scale.

**Honest footnote — variance shrinkage:** every synthetic std is slightly *below* its real
counterpart (Age 10.779<11.056, Income 30,738<31,665, Education 0.996<1.067, CreditScore
78.411<82.138). This is the known SMOTE interior bias — interpolation pulls samples toward
cluster centroids — and is pre-registered in `synthetic.http` test 3. Means this run skew
marginally *high* (≤ +1.5% on the continuous columns); the documented suite run skewed
marginally low. Both are normal request-to-request sampling variation, not defects.

### 2.2 Structural guarantees — zero violations in 200 rows

- **Convex hull (no extrapolation):** `syntheticStats` (row-derived extremes) sit strictly
  inside the seed bounds — Age [21.135, 59.84] ⊂ [21, 60]; Income [26,512.506, 129,795.769]
  ⊂ [26,000, 130,000]; CreditScore [551.512, 816.397] ⊂ [550, 820]; Education [0, 3]. **0
  hull violations.** No negative incomes, no impossible ages — the exact failure mode that
  disqualified CTGAN.
- **Categorical integrity:** every Education value across the 200 rows is exactly 0, 1, 2, or
  3 — not one float-noise value (the Copula failure mode). The schema's Categorical flag
  demonstrably reached the generator.
- **Plausibility (joint consistency):** each row is a blend of two neighboring real rows, so
  no chimeras — e.g. row 1 `[25.91, 31,202, 0, 593.5]` is a coherent young / low-income /
  HighSchool / low-score profile; row 7 `[49.55, 109,921, 3, 810.3]` a coherent senior /
  high-income / PhD / high-score one. No "Age 22 with Income 120k" combinations occur.
- **Self-reporting consistency:** the response's reported `syntheticStats` extremes and means
  are consistent with the visible rows; `ComputeStats`/`PearsonCorrelation`
  (SyntheticController.cs:207-257) behave correctly for this artifact.

### 2.3 Request contract

`generatedSamples: 200` with 200 rows matches the clamp contract
`Math.Clamp(request.NumSamples ?? 20, 1, 200)` (SyntheticController.cs:284). Whether the
caller asked for exactly 200 or was silently capped from a larger number is not recoverable
from the response — see gap 3.2.

---

## 3. Gaps, inconsistencies, and incorrect behaviors

### 3.1 "Synthetic" here means interpolation, not generation (interpretive, significant for adoption)
Every row is a convex blend of two real customer records (K=5 neighbors, gap λ ∈ (0,1)):
- **No novelty:** the generator can never produce a profile outside the seed envelope — fine
  for class balancing / augmentation, wrong for stress-testing or tail-risk scenarios.
- **Not privacy-safe by itself:** zero exact duplicates ≠ anonymization. Each synthetic row
  lies on a line segment between two real records; with auxiliary knowledge, attribute
  disclosure by linkage is feasible. If "synthetic data" is ever pitched as a privacy
  mechanism, this POC does not support that claim.

### 3.2 Silent input clamping (design trade-off, documented)
Out-of-range `numSamples` (0, −5, 100000) is corrected, never rejected — the endpoint returns
400 only for model-binding failures (wrong JSON type, empty body). A consumer cannot
distinguish "I asked for 200" from "my 100,000 was capped". Verified in `synthetic.http`
tests 7-8; worth an explicit `clamped: true` flag (or 422) if this graduates.

### 3.3 Statistics are reported, not asserted (minor)
The response computes `realStats`/`syntheticStats`/`correlations` but nothing checks them — a
regression (e.g. a library update breaking correlation preservation) would still ship a green
200 with bad numbers inside. The `.http` suite's expectations are the only guard and require a
human reader.

### 3.4 Reporting quirks (cosmetic, shared with the other two POCs)
- `timings.totalMs: 9.859` < `trainMs: 13.704` — `trainMs` is the cached one-time fit cost,
  not part of this request. Same naming confusion as Classification/NER.
- `system.libraryVersion: "0.204.0+…"` — known stale assembly attribute in the AiDotNet
  0.213.3 package (flagged by code comment, SyntheticController.cs:314-316).
- `columnSchema` serializes `categories: []` (empty array) for continuous columns where
  `null` would better express "not applicable" — trivial.

### 3.5 Ticket-side gap
"Findings are presented to the team for review" — process step, not verifiable from the repo.
Material is ready.

---

## 4. Recommended improvements (if the POC graduates)

1. **State the augmentation-vs-anonymization boundary in the API docs:** suitable for class
   balancing and dataset augmentation, not privacy-preserving data release. One paragraph
   prevents the most likely misuse.
2. **Add a fidelity assertion mode:** an optional flag / harness check that fails when any
   |real − synthetic| correlation gap exceeds a tolerance (e.g. 0.05) or any value escapes
   the hull — turning the response's self-evaluation into a regression gate.
3. **Make clamping visible:** include `requestedSamples` alongside `generatedSamples` (or
   return 422 for out-of-range input) so silent correction is auditable.
4. **Report distance-to-nearest-seed** per batch (min/mean) as a cheap privacy metric — "0
   exact duplicates" is weaker than it sounds; nearest-neighbor distance quantifies how close
   the blends sit to real records.
5. **Rename/annotate timing fields** (`trainMs` → `oneTimeFitMs`) — same fix as the other two
   POCs.
6. **Close the loop on the ticket:** present these findings, including the verified comparative
   results (SMOTENC ✓, CTGAN ✗ diverges, Copula ✗ degrades), as the team-review deliverable.

---

## Bottom line

The artifact does exactly what the implementation promises and the implementation does
exactly what the ticket asks: a working, configured, documented synthetic-data POC with
captured sample I/O, which both proves the library's working path (SMOTENC preserves marginals
within ~1.5% and **all six correlations within 0.01** at 4× oversampling, with perfect
structural integrity in 200/200 rows — convex hull respected, categoricals exact) and
documents its broken ones (CTGAN, Copula) → **PASS**. The open questions are about what
SMOTE-style synthesis *means* for downstream use — novelty and privacy limits — and those
belong in the team presentation, not in a bug tracker.
