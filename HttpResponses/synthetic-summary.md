# Synthetic Data POC — Ticket Compliance Analysis

**Artifacts reviewed:**
- `HttpResponses/response-synthetic-predict-text.json` (output of `POST /api/Synthetic/Generate`, 200 rows)
- `Controllers/SyntheticController.cs` (facade, SMOTE-NC generator, stats/correlation reporting)
- `ticket.md` (POC user story and acceptance criteria)

All numeric claims below were re-verified directly against the JSON artifact
(independent recomputation of stats, hull check, categorical check, seed-duplicate
check over all 200 rows) — not taken from the response's own self-reporting.

---

## Verdict: **PASS** — the cleanest of the three POCs

The ticket asks for a working, documented synthetic-generation POC that verifies
AiDotNet's claimed capabilities. This artifact satisfies every verifiable acceptance
criterion, and unlike the Classification and NER POCs there is no behavioral asterisk:
all structural guarantees the implementation promises (convex hull, exact
categoricals, correlation preservation) hold on every one of the 200 generated rows.
The caveats are interpretive (what SMOTE fidelity does and doesn't mean), not defects.

---

## 1. Ticket requirements vs. delivery

| Ticket requirement | Status | Evidence |
|---|---|---|
| POC created for Synthetic generation model type | ✅ Met | `SMOTENCGenerator<double>` via the library's first-class `ISyntheticTabularGenerator` API (`Fit(Matrix, ColumnMetadata, epochs)` → `Generate(n)`), behind `ISyntheticDataFacade` |
| Includes setup | ✅ Met | One-time lazy fit on 50 hardcoded customer rows (`FitGenerator()`, SyntheticController.cs ~line 136); singleton, `trainMs: 15.3` cached for all requests |
| Includes configuration | ✅ Met | `modelInfo` echoes K=5, Seed=42, 50 seed rows, and the full column schema with Education declared Categorical (HighSchool/Bachelor/Master/PhD) — the schema is actually passed to `Fit`, which is the "NC" in SMOTE-NC |
| Working test scenario | ✅ Met | `HttpRequests/synthetic.http` — 10 requests (fidelity audit, boundary+privacy audit, degenerate n=1, RNG behavior, clamping, validation) with verified expected outcomes; this JSON is the max-scale (200-row) scenario |
| Results documented | ✅ Met | Expected outcomes documented per request with measured values; response carries its own evaluation (realStats vs syntheticStats vs correlations) |
| Sample input/output captured | ✅ Met | This JSON is the captured sample I/O — 200 generated rows plus the full fidelity report |
| Findings presented to team | ⚠️ Not verifiable from artifacts | Process step; this document is the presentation-ready material |
| Verify library delivers on claimed capabilities | ✅ Met (nuanced) | SMOTENC delivers. The code documents what does NOT (verified on 0.213.3, comments ~lines 46–61): CTGAN diverges on small data (worse with more epochs), Copula degrades correlations to 0.82–0.93 and leaks float noise into categoricals — the comparative verification the ticket asked for |

---

## 2. Expected vs. actual behavior — every structural guarantee verified on this artifact

### 2.1 Statistical fidelity (the headline metric) — excellent

| Column | Real mean / std | Synthetic mean / std | Assessment |
|---|---|---|---|
| Age | 38.2 / 11.056 | 37.167 / 10.92 | mean −2.7%, std slightly below real |
| Income | 72,500 / 31,665 | 69,319 / 31,323 | mean −4.4%, std slightly below real |
| Education | 1.68 / 1.067 | 1.59 / 1.04 | consistent with the above |
| CreditScore | 706.84 / 82.138 | 699.44 / 81.841 | mean −1.0% |

All six pairwise correlations are preserved within **0.005** of real — e.g.
Age~Income 0.9882 vs 0.9893, Income~CreditScore 0.9689 vs 0.9681, Age~CreditScore
actually 0.9542 vs 0.9535. This is the property that justified choosing SMOTENC over
CTGAN and Copula, and the artifact confirms it at the maximum allowed scale.

Two honest footnotes: (a) every synthetic std is slightly *below* its real
counterpart — the known SMOTE variance-shrink bias (interpolation pulls toward
cluster interiors), exactly as pre-registered in `synthetic.http` test 3; (b) this
run's mean deviations (up to −4.4% on Income) are a bit larger than the ~1% measured
in the suite's documented run — sampling noise in which seed neighborhoods got
oversampled (this draw skewed slightly low: Education distribution 36/58/58/48 for
0/1/2/3 vs the seed's balanced 12–13 each ≈ 24%/26%/26%/24% proportions). Normal
request-to-request variation, not a defect.

### 2.2 Structural guarantees — zero violations in 200 rows (independently recomputed)

- **Convex hull (no extrapolation)**: 0 rows outside the seed bounds. Observed ranges
  Age [21.21, 60.00], Income [26,027, 128,943], CreditScore [551.06, 816.34] — all
  strictly inside the seed's [21, 60] / [26,000, 130,000] / [550, 820]. No negative
  incomes, no impossible ages — the exact failure mode that disqualified CTGAN.
- **Categorical integrity**: all 200 Education values are exactly 0, 1, 2, or 3 — not
  one float-noise value (the Copula failure mode). The schema's Categorical flag
  demonstrably reached the generator.
- **Privacy probe**: 0 of 200 rows exactly duplicate a seed row.
- **Plausibility**: joint consistency holds row by row — e.g. row 1
  `[39.51, 81,984, 2, 731.19]` is a mid-career Master's profile; there are no
  young-age/high-income chimeras, because SMOTE can only blend neighboring real rows.
- **Self-reporting is trustworthy**: the response's `syntheticStats` match my
  independent recomputation from the raw rows to the third decimal — the controller's
  `ComputeStats`/`PearsonCorrelation` code (~lines 196–246) is correct for this
  artifact.

### 2.3 Request contract

`generatedSamples: 200` with 200 rows delivered matches the controller's clamp
contract (`Math.Clamp(request.NumSamples ?? 20, 1, 200)`, ~line 273). Whether the
caller asked for exactly 200 or was silently capped from a larger number is not
recoverable from the response — see gap 3.3.

---

## 3. Gaps, inconsistencies, and incorrect behaviors

### 3.1 "Synthetic" here means interpolation, not generation (interpretive, significant for adoption)
Every row is a convex blend of two real customer records (K=5 neighbors, gap λ ∈
(0,1)). Consequences the team should weigh:
- **No novelty**: the generator can never produce a profile outside the seed
  envelope — fine for class balancing/augmentation, wrong for stress-testing or
  tail-risk scenarios.
- **Not privacy-safe by itself**: zero exact duplicates ≠ anonymization. Each
  synthetic row lies on a line segment between two real records; with auxiliary
  knowledge, attribute disclosure by linkage is feasible. If "synthetic data" is ever
  pitched as a privacy mechanism, this POC does not support that claim.

### 3.2 Silent input clamping (design trade-off, documented)
Out-of-range `numSamples` (0, −5, 100000) is corrected, never rejected — the endpoint
returns 400 only for model-binding failures (wrong JSON type, empty body). An API
consumer cannot distinguish "I asked for 200" from "my 100,000 was capped". Verified
behavior, documented in `synthetic.http` tests 7–8, but worth an explicit `clamped:
true` (or 422) if this graduates.

### 3.3 Statistics are reported, not asserted (minor)
The response computes realStats/syntheticStats/correlations but nothing checks them —
a regression (say, a library update breaking correlation preservation) would ship a
green 200 with bad numbers inside. The `.http` suite's expectations are the only
guard, and they require a human reader.

### 3.4 Reporting quirks (cosmetic, shared with the other two POCs)
- `timings.totalMs: 8.193` < `timings.trainMs: 15.3` — `trainMs` is the cached
  one-time fit cost, not part of this request. Same naming confusion as
  Classification/NER.
- `system.libraryVersion: "0.204.0+…"` — the known stale assembly attribute in the
  AiDotNet 0.213.3 package (flagged by code comment, ~line 303).
- `columnSchema` serializes `categories: []` (empty array) for continuous columns
  where `null` would better express "not applicable" — trivial.

### 3.5 Ticket-side gap
"Findings are presented to the team for review" — process step, not verifiable from
the repo. Material is ready.

---

## 4. Recommended improvements (if the POC graduates)

1. **State the augmentation-vs-anonymization boundary in the API docs**: this
   generator is suitable for class balancing and dataset augmentation, not for
   privacy-preserving data release. One paragraph prevents the most likely misuse.
2. **Add a fidelity assertion mode**: an optional request flag (or test harness) that
   fails when any |real − synthetic| correlation gap exceeds a tolerance (e.g. 0.05)
   or any value escapes the hull — turning the response's self-evaluation into a
   regression gate.
3. **Make clamping visible**: include `requestedSamples` alongside `generatedSamples`
   (or return 422 for out-of-range input) so silent correction is at least auditable.
4. **Report distance-to-nearest-seed** per batch (min/mean) as a cheap privacy
   metric — "0 exact duplicates" is true but weaker than it sounds; nearest-neighbor
   distance quantifies how close the blends sit to real records.
5. **Rename/annotate timing fields** (`trainMs` → `oneTimeFitMs`) — same fix as the
   other two POCs.
6. **Close the loop on the ticket**: present these findings, including the verified
   comparative results (SMOTENC ✓, CTGAN ✗ diverges, Copula ✗ degrades) as the
   team-review deliverable.

---

## Bottom line

The artifact does exactly what the implementation promises and the implementation
does exactly what the ticket asks: a working, configured, documented synthetic-data
POC with captured sample I/O, which both proves the library's working path (SMOTENC
preserves marginals within a few percent and correlations within 0.005 at 4×
oversampling, with perfect structural integrity in 200/200 rows) and documents its
broken ones (CTGAN, Copula) → **PASS**. The open questions are about what
SMOTE-style synthesis *means* for downstream use — novelty and privacy limits — and
those belong in the team presentation, not in a bug tracker.
