# Response to internal review, round 1

The review (`review_round1.md`) recommended major revision. Each action item is listed below with how it was handled.

1. **Event-level inference.** Added paired Wilcoxon signed-rank tests over the nine events (`src/15_revision_analyses.py`, `results/rev_stats.json`).
   - HiPhyBoost vs PC-GBM: mean ΔAUC +0.016, better in 6/9 events, p = 0.16 (not significant).
   - HiPhyBoost vs GBM: p = 0.07. HiPhyBoost vs global LR: p = 0.004.
   - The Abstract, Results, Table 3 caption and Conclusions now state that the gain is modest and not significant across events.
   - Tree budget: the ablation without offsets uses the same 700 trees and equals PC-GBM (+0.001), so the larger budget does not explain the gain.
   - Repeating training with different seeds was not done (compute cost).
2. **Leakage in φ and σ_u.** Disclosed in Methods and in the Limitations.
   - Added a sensitivity analysis with fixed φ = 15, 30 and 100. Coverage depends on φ; point accuracy hardly does.
   - The earlier claim of insensitivity was removed.
   - Full nested refits were not feasible (about 20 h of compute).
3. **Fair few-shot baselines and per-event results.**
   - Added PC-GBM with the same empirical-Bayes quasi-posterior. Results: the Bayesian update and the model contribute about equally at k = 1, and HiPhyBoost is more accurate at all k. The baseline covers the truth more often.
   - Per-event results added as Supplementary Table S2, including the Denali failure.
   - A negative-binomial tile model was not added. The Gaussian tile random effect is reported in Supplementary Section S4.
4. **Selection term.** Removed from the title and Abstract. It is described as optional, and its negligible effect under the quasi-likelihood is stated. Its large effect under the random-effect formulation is reported.
5. **Zero-shot forecast.**
   - The Bayesian predictive mean (median E/O 3.7) is now disclosed, and the plug-in/median forecast is used as the point forecast.
   - Added a Student-t predictive interval that accounts for estimating μ and τ (coverage still 7/9, median width ×850).
   - "Honest" wording toned down.
6. **Offset interpretation.**
   - "Geoscientific quantity" reworded to "landslide propensity".
   - Confounding is now stated explicitly: domain definition, polygon-to-point reduction, ShakeMap version and mapping completeness.
   - Interpretations are flagged as post hoc.
   - The Kaikōura completeness argument was corrected to match its fitted offset (−0.12).
7. **Numerical mismatches.** All fixed:
   - φ range: 7–193 with each event's own offset; 47–68 as actually applied.
   - Cinchona PGA is a mean, not a median.
   - Table 3 bolds the best AUC in each row.
   - Median AP lift of GBM and LR (3.9) now reported.
   - k = 40 excludes Niigata (stated).
   - Kaikōura statement corrected.
   - Abstract no longer quotes 26–43%.
8. **Baselines.** Noted that the Jibson curve is in-sample for Northridge. The global LR remains a deliberately simple structural analogue, as stated in Methods.
9. **SHAP.** Added grouped attribution shares (terrain, ground motion, Newmark), a random-ranking null (95th percentile 0.09) and a note on collinearity.
10. **Length.**
    - Supporting information moved to a separate file (`wiley/supporting_information.tex`).
    - Main text cut from 24 to 20 pages. The reliability diagrams moved to SI, the global-LR subsection was condensed, and the few-shot figure lost its uninformative AUC panel.
    - The Abstract was cut to about 280 words.
    - The main figures were kept because the authors asked for extensive visualisation.
11. **Typesetting.**
    - Removed the placeholder box on the title page.
    - Fixed the equation overflow.
    - Fixed the Figure 8b legend.
    - Put the corresponding and submitting authors in the correspondence line.
    - The remaining 261 pt overfull warnings come from the Wiley class header; the template sample produces them too.
12. **Citations.** Sentences that rest on title-only references now say no more than the titles (Noto, Hualien, near-real-time, Taiwan cross-event, hierarchical Bayesian, frequency requirements). The Sassa (2005) citation is limited to the typhoon rainfall.
13. **NZ relevance.**
    - The low-shaking regime where HiPhyBoost does worse, and the Denali failure, are discussed explicitly.
    - An Alpine Fault scenario demonstration and frozen-model tests on Noto/Hualien 2024 are listed as future work. Their inventories and scenario ShakeMaps were not openly available in usable form.
