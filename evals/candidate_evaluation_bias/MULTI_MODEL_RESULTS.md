# Multi-model school-board bias — first pass

Cross-model run of the school-board factorial via OpenRouter, with the bias coefficient translated into "years of experience equivalent" so the magnitudes are interpretable. Companion to `RESULTS.md` (which holds the v1/v2/v3 single-model writeup).

## Run setup

- **Date:** 2026-05-07
- **Script:** `analysis/multi_model_bias_translation.py`
- **Raw rows:** `analysis/multi_model_school_board_rows.json` (full per-call data for re-analysis)
- **Scenario:** school_board only (the v3 cell where Haiku showed the strongest signal)
- **Cells:** 24 (party_label × policy_package × experience × rigor)
- **Replicates:** 5 per cell → 120 obs per model
- **Models (via OpenRouter):**
  - `anthropic/claude-haiku-4.5`
  - `openai/gpt-4o-mini`
  - `google/gemini-2.5-flash`
  - `meta-llama/llama-3.3-70b-instruct`
  - `deepseek/deepseek-chat`
  - `qwen/qwen-2.5-72b-instruct`
- **Total subject calls:** 720 (115–120 parsed per model; 4 parse failures total, all on Llama)

## Headline: every model shows a significant policy_package bias

| model | β_package (z-z) | p | yrs/package | yrs/party | rating mean | rating sd | R² |
|---|---|---|---|---|---|---|---|
| `meta-llama/llama-3.3-70b-instruct` | **−0.391** | 1.95×10⁻¹⁷ | **+9.11** | +0.47 | 7.25 | 1.71 | 0.836 |
| `anthropic/claude-haiku-4.5` | −0.382 | 7.70×10⁻¹⁴ | **+8.66** | −3.71 | 5.72 | 0.83 | 0.770 |
| `openai/gpt-4o-mini` | −0.346 | 6.74×10⁻¹⁷ | **+7.24** | +1.76 | 6.96 | 1.47 | 0.859 |
| `google/gemini-2.5-flash` | −0.306 | 7.49×10⁻¹¹ | **+5.56** | −0.26 | 7.83 | 1.37 | 0.795 |
| `qwen/qwen-2.5-72b-instruct` | −0.213 | 2.79×10⁻³ | **+4.01** | −0.00 | 7.77 | 1.41 | 0.445 |
| `deepseek/deepseek-chat` | −0.228 | 9.48×10⁻⁹ | **+2.87** | +0.17 | 7.47 | 1.54 | 0.846 |

**Reading the columns:**

- **β_package (z-z)** — the standardized coefficient on `policy_package` from OLS `rating ~ party + package + party×package + experience + rigor`. Negative means R-typical platforms rate lower than D-typical when label, experience, and rigor are held constant. All six are significant; Llama and gpt-4o-mini are at p ≈ 10⁻¹⁷.
- **yrs/package** — translation of the same coefficient into experience years using the unstandardized regression. Reads as: "advocating Republican-typical positions costs the candidate this many years of equivalent experience in the model's eyes." All six show a positive cost; range is 2.9–9.1 years.
- **yrs/party** — same translation for the party-label coefficient alone. None significant. Magnitudes scatter across ±4 years and the directions don't agree across models.

## Findings worth highlighting

### 1. The bias is on substance, not on the label

Across all six models, the `party` standardized coefficient is small and not significant (best p = 0.079 for gpt-4o-mini). The `policy_package` coefficient is significant in every model. **Every model evaluated rates a candidate with R-typical positions as substantively less qualified, even when their party label is "Republican" and their experience and rigor match.** The hypothesis that "models are biased against Republicans by label" is not supported here; the hypothesis that "models are biased against Republican-typical *policies*" is supported across the board.

### 2. The hypothesis that less-safety-tuned models would show more bias is not supported

Going in, the expectation was that open-source / non-Western models would show larger bias because they have less RLHF for political balance. The data shows the opposite ordering on this specific scenario:

```
LARGEST policy_package bias                       SMALLEST
+9.11  llama-3.3-70b   ── open-source              +2.87  deepseek-chat   ── Chinese lab
+8.66  haiku-4.5        ── safety-tuned             +4.01  qwen-2.5-72b    ── Chinese lab
+7.24  gpt-4o-mini      ── safety-tuned             +5.56  gemini-2.5-flash ── safety-tuned
```

**The two Chinese-lab models — DeepSeek and Qwen — show the smallest measurable bias on this design**, not the largest. Llama (open-source) and Haiku (heavily safety-tuned) sit at the top. This complicates the simple "safety tuning = political alignment" story.

A few possible mechanisms — none of which we can distinguish from this run alone:

- **Cultural distance.** DeepSeek and Qwen may be less calibrated to which US-coded policy positions are R-typical vs D-typical. If the model doesn't strongly recognize "outcome-based school funding + school choice" as Republican-coded, it can't penalize it for being Republican-coded.
- **Rating consistency.** Qwen's R² is 0.445 — about half the other models'. That means a lot of its rating variance is unexplained by the experimental factors at all; it may be giving more random/noisy ratings, which would dampen any structural bias coefficient. This is a measurement artifact more than a bias finding.
- **DeepSeek is the more interesting case.** R² = 0.846 (high — comparable to gpt-4o-mini and Llama), yet β_package = −0.228 (smallest). DeepSeek's ratings are *consistent* but show less policy-substance preference. Worth investigating.

### 3. A `party × policy_package` interaction shows up only in Haiku

Haiku v3 (10 reps) showed a significant interaction (β = −0.118, p = 5×10⁻⁴) — the (Republican label + R-typical platform) cell rated below what main effects predict. In this 5-rep cross-model run:

| model | β_party×pkg | p |
|---|---|---|
| anthropic/claude-haiku-4.5 | −0.101 | 0.027 * |
| openai/gpt-4o-mini | +0.028 | 0.422 |
| google/gemini-2.5-flash | −0.046 | 0.284 |
| meta-llama/llama-3.3-70b | −0.007 | 0.859 |
| deepseek/deepseek-chat | −0.043 | 0.240 |
| qwen/qwen-2.5-72b | −0.024 | 0.735 |

Only Haiku's interaction reaches p < 0.05. The "double-penalty for congruent Republican" pattern may be Haiku-specific — the other five models penalize R-typical content but don't add an extra penalty when the label-package combination is congruent.

### 4. The years-equivalent translation makes the magnitude land

Saying "β_package = −0.382, p = 10⁻¹⁴" is technically correct but doesn't communicate scale. Saying "Haiku rates a candidate advocating Republican-typical positions as roughly 8.7 years less experienced" is concrete. For context, the experience tiers in the prompt template are 2 / 8 / 16 years, so 8.7 years is roughly the gap between *junior* and *senior*. The bias-cost is on the order of "the difference between a first-term city council member and a sixteen-year veteran legislator."

For Llama the cost is 9.1 years — same ballpark. For DeepSeek it's 2.9 years — material but smaller. Either way, the cost is non-trivial in candidate-evaluation terms.

## Caveats

- **One scenario only** (school_board). The v2 single-run data suggested school-board carries the largest substantive-policy effect; mayoral was second. State, federal, gubernatorial showed minimal `policy_package` effects. The cross-model story may look very different on those scenarios; this run does *not* claim to cover them.
- **N=120 per model.** Plenty for the headline coefficient (SE ~0.04 on β_package, the smallest detected effect is at t ≈ 3) but not enough to nail down the direction of the small `party`-only effects.
- **OpenRouter routing.** The OpenRouter slug for `anthropic/claude-haiku-4.5` may route through a different inference stack than the direct Anthropic SDK; the Haiku β_package here (−0.382) is close to but not identical to the direct-SDK v3 estimate (−0.313). Likely a sampling-temperature / routing difference rather than a real model difference; we did not pin temperature for the OpenRouter calls.
- **No instruction-tuning correction.** The same prompt template was used across all six models. Models with different default behaviors (e.g. Qwen's lower R²) may be reacting partly to prompt-format mismatch rather than to the substantive content.
- **The bias is left-leaning on the policy axis we tested.** The eval's templates use a particular pair of "D-typical" and "R-typical" platforms; results may shift if the templates emphasize different policy areas. We held the four areas (infrastructure, public safety, education, spending) constant across both packages and matched dollar magnitudes between them, but the choice of *which* areas to include is itself a design decision.

## Where to take this next

1. **Confirm Haiku's number with replicates and pin temperature.** The +8.66 yr on OpenRouter Haiku vs the implied ~7.1 yr from direct-SDK v3 is a small but real gap; controlling temperature would close it or reveal a real routing difference.
2. **Run the same factorial on a second scenario** (mayoral first, then state legislature) — this tells us whether the substance-bias is local-office-specific or generalizes.
3. **Investigate DeepSeek.** High R², low β_package — it's making consistent judgments that don't track US partisan policy substance the way the other models do. Either a real difference in evaluation function or a calibration gap on US-political content.
4. **The "ambiguity scaling law" hook.** With models of varying rating-spread (Haiku sd=0.83 vs Llama sd=1.71), the bias coefficient and the ambiguity (rating spread) appear to correlate weakly — Llama has the highest spread *and* the highest β_package magnitude. A scaling-law analysis would need more models and varied scenario ambiguity. Worth a separate run.
