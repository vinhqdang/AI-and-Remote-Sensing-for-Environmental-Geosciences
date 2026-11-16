# Response to internal review, round 2

The review (`review_round2.md`) recommended a light major revision. Its numerical audit found all 30 recomputed numbers correct. The changes made in response are listed below.

1. **Subsets and dependence between folds.**
   - Added a bootstrap 95% interval of the mean AUC gain (0.006–0.020) and a sign test (p = 0.002), and stated that fold dependence makes the p-values approximate.
   - Now reported: 12/16 for the new Pacific Rim events (p = 0.07) and 9/13 outside the Pacific (p = 0.15).
   - The Abstract no longer quotes "20 of 25 Pacific Rim", because that count includes the development events.
   - Script: `src/24_offset_regression.py`, `extras`. Output: `results/global/global_auc_subsets.json`.
2. **Nine-event vs 38-event results.**
   - Methods now explains that the training pool and sampling differ, and both sets of results are reported.
   - In the 38-event run HiPhyBoost wins 8/9 of the original events (p = 0.02).
   - Kaikōura is reported in both runs: E/O 1.06 vs 0.52, offset −0.12 vs +0.62.
3. **Covariate claims.**
   - Correlations now carry p-values, and the text states that none survives a Bonferroni correction.
   - The wetness claim was removed from the Abstract and Conclusions; wetness is reported as not significant (p = 0.11).
4. **Forecast-time legitimacy.** This drove the main redesign.
   - Covariates are now split into forecast-time (F1 shaking/ShakeMap including mechanism, F2 climate, F3 both) and post-event (P1 inventory, P2 plus domain area, P3 all).
   - The p5 polygon area, which was derived from the target inventory, was removed.
   - Result: F1 and F3 (forecast-time) narrow the intervals from ×364 to ×132–137 at unchanged or higher coverage. The predictive-density gain is +0.16–0.18 (paired p ≈ 0.04; permutation p = 0.005). Climate alone does not help.
   - The Abstract now reports only the forecast-time result.
5. **Domain confound.**
   - Domain area is reported as the strongest correlate (r = −0.53, p = 0.001).
   - Figure 11b now plots offsets against domain area.
   - The magnitude effect is discussed as partly a domain-size effect, and the Conclusions say that offsets measure mapping as much as propensity.
6. **Regression design.**
   - Removed the duplicated `p5_missing` flag.
   - Extended the λ grid to 10⁴.
   - Added mechanism to the shaking set.
   - Added a placebo with 200 random permutations of the covariate vectors (Supplementary Table S5).
   - The "pre-specified" wording was replaced by "evaluated only on held-out events, with a permutation reference".
7. **Kaikōura.** It is covered by every conditional prior in the revised analysis (F3 interval 0.04–4.5 × observed). The New Zealand section was rewritten with both sets of numbers.
8. **Data quality.**
   - The station-count limitations and the outcome-defined area used for TerraClimate averaging are stated in Methods, together with the operational alternative (strongly shaken area).
   - Events with few landslides are visible in Table S4.
9. **Inconsistencies.**
   - Conclusion 3 now gives both ranges (5.7 and 7.1 log-odds units); Conclusion 4 gives both coverages (7/9 and 33/38).
   - Title changed to "…across the Pacific Rim and worldwide…".
   - The poor transfer to Tohoku-Oki and Gansu is discussed.
10. **Citations and data.**
    - Table S4 lists the ScienceBase identifier of every inventory.
    - The Data Availability statement now includes ComCat, TerraClimate via Planetary Computer, and the use of Colab GPUs.
11. **Presentation.**
    - The global-extension Results section moved to the end of Results, and Experiment 4 now follows Experiment 3 in Methods.
    - Figure 11 was revised: panel (a) labelled, panel (b) shows domain area, panel (c) separates forecast-time from post-event covariates.
    - The remaining 261 pt overfull warnings come from the Wiley class itself.
    - Title-page metadata (volume, DOI, dates) is filled in by the journal at publication.

Follow-up (both items previously listed as not done):
- **Climate over the shaken area.** The TerraClimate covariates are now averaged over the strongly shaken area: up to 300 ShakeMap nodes with PGA ≥ 0.2 g (`src/28_climate_shaken_area.py`). They are therefore forecast-time and no longer outcome-defined. The landslide-area versions are kept as a sensitivity set.
  - The values correlate 0.95–0.98 with the landslide-area versions.
  - The correlation between antecedent precipitation and the offset rises to r = 0.37 (p = 0.02).
  - The full forecast-time prior (F3) narrows intervals to ×107 (from ×364) with 34/38 coverage. Its LPD gain is +0.21 (paired p = 0.03; permutation p = 0.005).
- **Confidence intervals in Figure 11c.** Bootstrap 95% intervals over events for the median width, and Wilson 95% intervals for coverage. Coverage differences between priors are not significant (the text says so); the reduction in width is.
