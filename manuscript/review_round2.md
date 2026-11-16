# Referee report, round 2
Manuscript: "Transferable coseismic landslide forecasting across the Pacific Rim with hierarchical physics-anchored boosting and overdispersion-aware Bayesian updating" (NZ J. Geol. Geophys., special issue "AI and Remote Sensing for Environmental Geosciences").
Scope: main.pdf (21 pp.), supporting_information.pdf (5 pp.), LaTeX sources, src/20-27 and 15, results/, data/global. Numbers were recomputed with pandas (and the ridge/offset regression re-run in a scratch directory). Nothing in the repository was edited, committed or pushed.

## 1. Summary and recommendation

**What changed.** The authors addressed round 1 with event-level Wilcoxon tests, disclosure of leakage, fairer few-shot baselines, a Student-t zero-shot interval, softened offset language and a separate SI. The major addition is a 38-earthquake extension (9 original + 29 new inventories from the USGS repository).
- HiPhyBoost vs PC-GBM: better in 29/38 events (Wilcoxon p = 0.0006).
- Offsets are regressed on event covariates (inventory type, mapping extent, ShakeMap stations, TerraClimate wetness, Mw, depth, mechanism).
- A covariate-conditioned zero-shot prior narrows the interval from x364 to x128.

**Assessment.** The extension is the most valuable change. It answers my round-1 objection that n = 9 cannot support the transfer claim, and the headline AUC numbers reproduce exactly. The effect is, however, small: median AUC difference +0.011, mean +0.013, bootstrap 95% CI of the mean about 0.006-0.020. 8 of 38 events have |dAUC| < 0.005 and 9 of 38 are negative. The p-value is also optimistic because the 38 LOEO folds share 36 of 37 training events.

The offset-covariate part is much weaker than the abstract and Conclusions imply.
- Most covariate correlations are not significant. The "wetter months" claim, which is in the abstract, is not supported (r = 0.26, p = 0.11, and the climate set has a *negative* predictive-density gain).
- The only covariate set that helps (inventory/mapping) is not knowable before an earthquake.
- The covariate-conditioned prior misses Kaikoura, the paper's New Zealand anchor case (E/O interval 0.008-0.75; median 0.08).
- The New Zealand discussion and Conclusions still rest on nine-event numbers that the 38-event results contradict.

**Recommendation: MINOR-TO-MAJOR; I recommend MAJOR REVISION (light).** The core claims on transfer are now defensible. The covariate and zero-shot claims, the NZ section and the abstract need re-scoping, and several internal inconsistencies must be fixed. No new large compute is required. The needed analyses are cheap re-uses of the saved scores.

## 2. Round-1 action items

| # | Item | Status | Reason |
|---|---|---|---|
| 1 | Event-level inference | Adequately (n=9) / Partly overall | Wilcoxon reported; the 38-event test is the real fix. Folds are dependent, seeds/tree budget not repeated, and the 9-event result (6/9, p = 0.16) and the 38-event result for the same nine events (8/9, p = 0.02) are not reconciled. |
| 2 | Leakage in phi/sigma_u | Partly | Disclosed and phi sensitivity added; no nested fit (acceptable given compute), but Abstract and Conclusions still say "after one to ten tiles ... median factor 1.4-1.6" without the caveat. |
| 3 | Fair few-shot baselines, per-event results | Adequately | PC-GBM with the same EB quasi-posterior and Table S2 added. No count model (stated). |
| 4 | Selection term | Adequately | Removed from title and abstract. |
| 5 | Zero-shot definition | Adequately | Bayes mean disclosed (E/O 3.7), t-interval added. Still no proper score in the main text (I computed one; see 3.3). |
| 6 | Offset interpretation | Partly | Wording softened in Methods/Discussion, but the new abstract says offsets "are lower for point-catalogue inventories and higher after wetter-than-normal months" (not significant), and Conclusion 3 keeps the 9-event narrative. Domain-definition sensitivity was not tested; I find the offset correlates more strongly with log domain area (r = -0.53, p = 0.0006) than with any covariate. |
| 7 | Numerical mismatches | Adequately | Spot-checked phi, PGA mean, Table 3 bolding, k=40: fixed. |
| 8 | Baselines | Partly | Northridge in-sample note added. Global LR still unregularised/4 predictors. |
| 9 | SHAP | Adequately | Grouped shares and null added. |
| 10 | Length | Partly | 21 pp. main + 5 SI; abstract 288 words (OK). The added section/figure/Methods paragraph offset much of the cut. |
| 11 | Typesetting | Mostly | Placeholder box gone. The overfull warnings (22, all 261.8 pt "while \output is active") remain and are attributed to the class; the title page still has empty Received/Revised/Accepted, DOI 10.1002/0000, volume 0. |
| 12 | Citations | Partly | Title-only references toned down; the 29 new inventories, TerraClimate, ComCat and Planetary Computer are not credited (section 5). |
| 13 | NZ relevance | Partly | Low-shaking and Denali discussion added; Alpine Fault scenario and Noto/Hualien frozen test still "future work". The NZ section was not updated for the 38-event results (section 5). |

## 3. Numerical audit of new and changed content

### 3.1 Verified correct (recomputed; 30 items)
1. 38 events, 25 Pacific / 13 other (global_auc.csv): OK.
2. HiPhy better than PC-GBM in 29/38: OK.
3. Median AUC 0.795 vs 0.763 (0.7953, 0.7628): OK.
4. Mean difference +0.013 (0.01284): OK.
5. Wilcoxon p = 0.0006 (0.000568), abstract p < 0.001: OK.
6. Pacific 20/25, p = 0.0015 (0.00145): OK.
7. Non-Pacific 9/13, p = 0.15 (0.146): OK.
8. 29 new events 21/29, p = 0.014 (0.0138): OK.
9. Kaikoura 0.917 vs 0.892 (0.9174 / 0.8919): OK.
10. Median E/O error factor 2.9 (2.95) for HiPhy and 4.0 (3.96) for PC-GBM: OK.
11. Offsets from -4.2 (Southern Italy, -4.18) to +2.9 (Cinchona, +2.88): OK.
12. tau = 1.73 (1.7296): OK.
13. r point inventory = -0.42 (-0.418): OK.
14. r mapping extent = +0.30 (0.300): OK.
15. r Mw = -0.42 (-0.423): OK.
16. r normal faulting = -0.39 (-0.394): OK.
17. r antecedent precipitation = +0.26 (0.261): OK.
18. r mean annual precipitation = +0.24 (0.235): OK.
19. r stations = -0.13 (-0.132): OK.
20. M0 width 364 (363.7), coverage 0.87 (33/38): OK.
21. M4 width 128 (128.4), coverage 0.95 (36/38): OK.
22. M1 width 180 (179.8), coverage 0.92 (35/38): OK.
23. LPD gain 0.17 nats (0.1657): OK.
24. LPD gain p = 0.08 (0.0767): OK.
25. Median point-error factor 3.9 (M4, 3.92) vs 2.9 (M0): OK.
26. "Almost threefold" (364/128 = 2.83): OK.
27. Table S4 (38 rows x 8 columns: Mw, cells, km2, inventory, stations, rain ratio, both AUCs, b_e): all match the source files.
28. Abstract is 288 words: OK.
29. 29 new = 16 Pacific + 13 other; 1971-2019; the 13 non-Pacific are Mediterranean (5), Himalaya (2), China (5), Caribbean (1): OK.
30. The within-text 9-event numbers (p = 0.16, 0.07, 0.004, 700 trees): unchanged and consistent with rev_stats.json.

### 3.2 Mismatches, imprecisions and unreported results
- **A. "21 of 29 events not used in its development" and "20 of 25 Pacific Rim" are mixtures.** The Pacific 25 contain the 9 development events (8/9 better). Among the *new Pacific* events the result is 12/16, p = 0.065, which is not significant. The 13 non-Pacific events are 9/13, p = 0.15. Say this explicitly; the abstract implies the Pacific effect is established independently of development.
- **B. The nine original events in the 38-event run.** HiPhy is better in 8/9 (p = 0.0195; median dAUC +0.025; HiPhy 0.827 vs PC-GBM 0.809), whereas the 9-event analysis says 6/9, p = 0.16. Aisen flips from -0.026 to +0.037 and Cinchona is -0.014 to -0.003. PC-GBM Aisen changes 0.760 to 0.654 just by adding events. The paper presents both analyses without comment. Table 3 (9-event) and Table S4 are therefore different LOEO experiments on the same targets.
- **C. Sign test is p = 0.0017** (not < 0.001) and the mean-difference bootstrap CI is 0.006-0.020. The text does not give an effect-size interval.
- **D. Significance of correlations (n = 38, 13 tested):**

| Covariate | r | p |
|---|---|---|
| point inventory | -0.418 | 0.009 |
| Mw | -0.423 | 0.008 |
| normal faulting | -0.394 | 0.014 |
| mapping extent | +0.300 | 0.067 |
| antecedent precipitation | +0.261 | 0.114 |
| mean annual precipitation | +0.235 | 0.156 |
| stations | -0.132 | 0.43 |

  No correlation survives Bonferroni (0.05/13 = 0.0038). The text says "in physically and observationally sensible directions" and the abstract names wetter months as a finding. The p5-area correlation (r = -0.415) is an imputation artefact (see 4.5).
- **E. Climate covariates do not help prediction.** M3 LPD gain is -0.018 (p = 0.096), width 339 vs 364, error 2.91 vs 2.95. M2 (shaking/ShakeMap): +0.012, p = 0.57. The text says only "change little"; the abstract and Conclusion 3 should not mention wetness as a result.
- **F. "Without loss of coverage" / "coverage rises from 0.87 to 0.95".** 33/38 vs 36/38; binomial CIs overlap fully (0.72-0.96 vs 0.82-0.99; neither differs from 0.90, p = 0.43/0.58). The difference is 3 events.
- **G. Kaikoura under the conditional prior.** M4 predicted b = -1.87 (oracle b* = +0.67), median E/O = 0.08, interval 0.008-0.75, so *not covered*; M1 median E/O 0.17; M0 median 0.52 (interval 0.026-9.4). M4 also misses Southern Italy. M0 misses Cinchona, Lombok, Pisco, San Miguel and Southern Italy. The NZ section quotes none of this.
- **H. Point forecasts of the original events change with the larger training set.** Kaikoura plug-in E/O is 1.06 (9 events) vs 0.52 (38 events); Iwate 1.12 vs 0.37. Only 13/38 events are within a factor of 2 for either HiPhy or PC-GBM. The NZ text ("within 6% of the inventory") is therefore contradicted by the paper's own larger analysis.
- **I. Offset range and identity of extremes.** Conclusion 3 still says "5.7 log-odds, from Denali (lowest) to Cinchona (highest)". Across 38 events the range is 7.06 (Southern Italy -4.18 to Cinchona +2.88; Denali is -2.48); e^7.06 = 1170. The discussion also says Kaikoura "is closest to the population mean (b = -0.12)" but its 38-event offset is +0.62.
- **J. Interval comparisons are a mix.** Section 4.6 reports x850 (t-interval, 8 training events); the 38-event baseline is x364 (36 d.o.f.). The widening is mostly a sample-size effect. Say "x364 with 37 events" next to "x850 with 8".
- **K. Interval score (my calculation, log scale, alpha = 0.1).** Mean 7.37 (M0), 6.16 (M1), 7.19 (M2), 7.41 (M3), 5.98 (M4); M4 is better in 35/38 events. This is a more informative summary than coverage and should be reported, but any p-value across folds is not meaningful (dependent folds).
- **L. Out-of-sample skill.** RMSE of predicted vs oracle offset: 1.80 (M0), 1.64 (M1), 1.51 (M4), against SD(b*) = 1.83. M4 explains about 32% of the variance, M1 about 19%. These are the numbers a reader needs and are not stated.
- **M. M1 and M4 LPD p-values are identical (0.0767) with different gains (0.090 vs 0.166).** Plausible (rank statistic), but worth a note in case of a copy error. The Wilcoxon p-values are two-sided, the text says "consistent in direction".
- **N. Station counts.** 12/38 events (32%) have 0 stations in Table S4, including Palu, Aisen, Cinchona, Kashmir, Lombok, Mesetas and Guatemala. For Sichuan-Yunnan-Guizhou, Lombok and Milin the grid.xml value is -1 (unknown) and the station-list parser returned nothing, yet they appear as 0 (see 4.6). The text says "weakly related (r = -0.13)" without caveat.
- **O. Table S4 caption** says offset "from the fit to all 38 events" (correct), but the Results figure uses all-event offsets while the regression targets use fold-wise oracle offsets b* (SD 1.83). These are different quantities (see 4.4).
- **P. Southern Italy has only 68 positive cells** (199 inventory points, 34% retained after gridding/DEM masking); Livermore 105, San Miguel 134, Central Italy 176. The selection rule is on inventory features (>= 100), but the effective positive count is lower. These events carry the extreme offsets (-4.18, -0.54, -2.96, -2.17) and weigh equally in event-balanced training.

### 3.3 Not stated but computed
Placebo test (20 random permutations of covariates across events, same pipeline): mean LPD gain of M4 vs M0 is -0.025 (range -0.062 to +0.026), against +0.166 observed. So the M4 gain is not pure overfitting noise; this is a point in the authors' favour and should be reported.

## 4. Methodological scrutiny of the global extension

**4.1 Event selection.** One inventory per event (most features), >= 100 features, ShakeMap available: all 29 selected events were built, none skipped (MAX_AREA 80 000 km2 not triggered; Wenchuan hull is 77 832 km2). Good. Concerns:
- Selection on the repository's coverage is itself biased towards well-studied events.
- Where several inventories exist, "most features" favours point catalogues or inventories with split polygons (feature count is not completeness).
- The rule is on features, not on mapped cells. See 3.2-P.

**4.2 Heterogeneity.** Inventories differ in type (point 13, polygon 25), mapping extent (6 published, 32 convex hulls), resolution (p5 area ranges 8 m2 to 4126 m2) and purpose (CEDIT catalogues vs systematic mapping). Polygon-to-point reduction removes size information. Domain definition is the biggest confounder. For 32/38 events the domain is the landslide convex hull + 2 km, an outcome-defined area. log10(domain area) correlates with offset at r = -0.53 (p = 0.0006), stronger than any covariate, and with Mw at r = +0.68. The reported "Mw effect" is therefore plausibly a domain-size effect (as the Results text half-concedes), and offsets estimated for hull-domains are not what a forecaster gets from a ShakeMap-defined domain. Please test (a) a forecast-time domain (e.g. PGA > 0.05 g or MMI >= VI, or a buffer around the 90% shaking contour) and (b) log area as a covariate.

**4.3 Harmonised negative sampling.** 150k negatives for all events with inverse-rate weights is fine for AUC/E/O. For the nine originals the data were subsampled from 250k, so their results differ from the 9-event analysis for a second reason besides the larger training set. Wenchuan has 197 160 positives against 150 000 sampled negatives, so the ratio is inverted for the largest event. State sampling-induced AUC noise (seed repeats) and confirm that results are invariant to NEG_MAX.

**4.4 Are the 29 new events independent of method development?** Partly. They were not used to choose the architecture, T = 3, round counts or monotone directions, and no hyperparameters were retuned; that is a fair claim. But (i) the nine development events are in every fold's training set, and the 29 new events share training sets with each other; (ii) the extension itself was run after seeing the nine-event outcome and the round-1 review; (iii) the event set, covariate sets and the Pacific-Rim flag (hand-drawn boxes: Mesetas and Lombok in, Wenchuan/Gansu/Milin/Nepal out) were defined by the authors, with no time-stamped pre-registration. "Pre-specified" in the Discussion cannot be verified; the git history shows one commit adding catalogue, covariates, regression and manuscript text together. I would call the 29 events an "independent test of the architecture" rather than a hold-out, and soften "pre-specified".

**4.5 Ridge offset regression.** n = 37 training events, up to 11 covariates (M4).
- Two columns are identical: `p5_missing == point_inventory` (r = 1.0), and `log_p5_area` is imputed with the median for point events. In M1/M4 the point indicator is double-counted by the ridge penalty, and the "p5 area" correlation (-0.415) is the same signal as the point flag.
- Lambda is chosen by LOO on a 7-point grid (0.1-100). M3 selects the upper edge 100 in 36/38 folds and M4 selects 10 in 37/38, so the grid is not exploring a sensible interior (M3 effectively says "no signal"). Extend the grid beyond 100.
- Student-t predictive uses dof = n - tr(H) and residual variance from the in-sample fit. This ignores the uncertainty from selecting lambda and from ridge bias. Empirically the predictive SDs are close to the realised RMSEs (M4: 1.44 vs 1.51; M0: 1.77 vs 1.80), which is reassuring, but coverage above nominal (0.95) for M4 deserves a comment.
- Targets are oracle offsets b* (transferred scores, matching the full total) in the evaluation but joint-fit offsets b_e in training. Document this.
- Forecast-time availability. `point_inventory`, `mapping_extent`, `p5_missing`, `log_p5_area` and `year` of the *inventory* are properties of the inventory that will be made after the earthquake (p5 area is computed from the target inventory polygons itself). They are not known before an earthquake. Using them is legitimate only as conditioning on a *chosen target definition* ("forecast what a complete polygon inventory would show"), not as an improved *zero-shot prior*. That is exactly the covariate set that provides the gain (M1: LPD +0.09, width 180). The forecast-time-valid sets M2 and M3 give nothing. The combination M2+M3 (and Mw, mechanism, which are not in any set but are among the strongest correlates) is not tested. M4's improvement should be reported as conditional on assumed inventory type, and a forecast-time-valid model (M2+M3+mechanism, or M1 with the inventory type set to the planned mapping protocol, e.g. complete polygons for NZ) should be reported separately. The Methods text is wrong that mechanism is among the regression covariates: mechanism, `rev` and `normal` enter no set although they are named in the Results.
- Multiple comparisons: 4 sets x 4 metrics, 13 correlations, post hoc subset tests (all, Pacific, non-Pacific, new-29). No adjustment; none of the LPD gains is significant at 5%.
- Dependence: LOEO folds share 36 of 37 training events, so fold-wise LPD/AUC differences are positively correlated and Wilcoxon p-values are anti-conservative. Use a permutation/bootstrap over events with the fold structure, or report effect-size CIs.

**4.6 ShakeMap station counts.** The covariate mixes (i) the grid.xml `seismic_stations` attribute, (ii) a station-list parser that counts non-macroseismic features (heuristic string match on station_type/source) and (iii) `fillna(0)` / `clip(lower=0)`. A "0" can mean "no stations", "Atlas/model-only" or "unknown". Station counts are also era-dependent (pre-1990 Atlas maps) and event-dependent (the DYFI/instrumental split and contributed versions vs Atlas). The "ShakeMap quality" interpretation in the Limitations is plausible, but the variable is too noisy to test it. Report unknowns as missing, add an indicator, or drop the variable.

**4.7 TerraClimate averaging.** Antecedent wetness is averaged over the bounding box of the 2nd-98th percentile of landslide-cell coordinates + 0.05 degrees (thinned to <= 400 grid cells). This area is defined from the outcome (landslide locations), so it is not available at forecast time and it varies with the mapped extent; for small events (Kobe, Lefkada, Niigata) it is a few 4-km cells. The preceding-month + elapsed-fraction precipitation is monthly and cannot resolve rainfall in the last days before the event. The ratio to climatology is unstable in dry seasons (San Fernando 0.23, Coalinga 2.37, Loma Prieta 2.41), so log ratio is dominated by dry-season months; an anomaly in mm (and soil moisture anomaly, which is computed but not used) would be better. Use a forecast-time area (e.g. ShakeMap MMI >= VI footprint or a fixed radius around the epicentre).

**4.8 AUC test.** Wilcoxon on 38 paired AUC differences is appropriate for the sign, but: dependence between folds (above); the effect (median +0.011) is small relative to between-seed variation that is never measured (no seeds repeated); the mean gain of 0.013 vs the nine-event 0.016 and the 8/9 vs 6/9 flip show instability; and the event-balanced weighting gives events with 68-300 positives the same weight as Wenchuan. Provide seed repeats for a subset (e.g. 5 seeds on all 38 events with the GPU run) and an event-level paired bootstrap CI.

## 5. Remaining overclaims, inconsistencies and citation issues

1. **Abstract.** "higher after wetter-than-normal months" and "conditioning ... narrowed zero-shot intervals almost threefold without loss of coverage": not supported as stated (3.2-D/E/F, 4.5). Also does not say that the point forecast did not improve (3.9 vs 2.9) and the paper's own text says the predictive-density gain is not significant (p = 0.08); the abstract omits this. Replace with "inventory type and ShakeMap coverage predict part of the offsets; intervals narrow when the inventory type is known".
2. **Conclusion 2** ("confirmed on 38 earthquakes") and **Conclusion 3** (5.7 log-odds, Denali/Cinchona, "wetter-than-normal months"); update with the 7.1 range and drop the wetness statement.
3. **Conclusion 4** (7/9 coverage, "factor of 20-50 in the worst cases") are 9-event statements, while the extension shows 33/38 and PC-GBM errors up to x17 (E/O 0.058 to 169, i.e. 2900-fold). Choose one reference set or give both.
4. **Section 5.4 (NZ).** "Kaikoura is the event closest to the population mean (b = -0.12)", "within 6%", "interval spans roughly two orders of magnitude", "a handful of tiles is enough" and "this is encouraging for forecasting an Alpine Fault earthquake" are 9-event statements. With 38 events Kaikoura has b = +0.62, E/O = 0.52 (factor 1.9 low) and a zero-shot interval of x361; the covariate prior excludes the observed total. The honest NZ message now is that covariate conditioning would *not* have helped for Kaikoura. The statement that complete GNS mapping "should raise Kaikoura's offset" is now contradicted by +0.62 being higher than the mean and the "point catalogue" covariate mispredicting Kaikoura, whose source data are in fact complete mapping.
5. **Discussion 5.1** says "inventory type and mapping extent are the strongest correlates" (Mw is as strong; mapping extent is p = 0.067) and that "antecedent wetness has the expected positive sign" (not significant). The sentence "we pre-specified the covariate sets and evaluated them only on held-out events" overstates (4.4).
6. **"Point catalogue" interpretation.** The text says point catalogues "record only a fraction of the landslides". Within the point group the offset is bimodal: Italy -1.2 to -4.2, Pisco -3.6, San Miguel -3.0 versus Kaikoura +0.62, Palu +0.47, Western Sichuan +0.66, Loma Prieta +0.21. Excluding the four Italian events r = -0.30 (p = 0.09). The effect is largely a CEDIT/Latin America cluster and has regional confounding (Mediterranean/normal faulting, 4 of the 8 normal-faulting events are the Italian catalogues). Mechanism "normal" is confounded with point inventory (r = 0.44).
7. **"Pacific Rim" is defined by 12 hand-drawn boxes** (src/20_global_catalogue.py); the paper should say so in Methods and list the criteria; Tohoku-Oki (M9.1, AUC 0.56/0.55) and Gansu (HiPhy -0.050) are the worst cases and are not discussed.
8. **Two LOEO analyses for the same nine targets** (3.2-B) without explanation of which is the "main" result. Since the abstract leads with "nine ... then 38", present the 38-event result as the primary result and demote the nine-event table, or reconcile in one sentence.
9. **Title** still says "across the Pacific Rim" while the main test set is global (13/38 outside the rim).
10. **Terminology.** "Zero-shot" is used for the median-offset forecast, the Bayesian mean and the covariate-conditioned forecast; "mapping extent" (published polygon) vs "domain" (hull) vs "study area"; "PC-GBM" definition.
11. **Citations.** The 29 new inventories are used without individual citations or DOIs (CEDIT, Xu et al., Valagussa, Harp, Wartman, Lacroix, Marc, Basharat, Ferrario, Delgado, etc.). The Data Availability statement says "DOIs in the reference list" but only covers the nine. TerraClimate (abatzoglou2018 only; no data DOI or Planetary Computer), ComCat/USGS event data and the Colab runs are not cited/mentioned in the Data Availability section. The round-1 title-only references remain as before.

## 6. Presentation and typesetting

- **Length.** 21 pp. main + 5 pp. SI, 17 100 words in main.pdf including references. Section 4.8 was added between few-shot (4.7) and attribution (4.9) in the Results, and Experiment 4 sits before Experiment 3 in the Methods. Figure 11 (p. 15) is three pages from its section opener (p. 12). Move the global extension to follow Section 4.4 (HiPhyBoost) or make it the primary result; reorder Experiments so they are numbered consecutively.
- **Figure 11.**
  - (a) Unlabelled points; label at least Tohoku-Oki, Gansu, Mesetas, Kaikoura.
  - (b) Legend overlaps data; the claimed wetness association is not visible (point catalogues are clustered at low x and low b); add a regression line with CI or drop the panel.
  - (c) Dual axis with a line connecting categorical bars, no uncertainty on coverage (add binomial CIs) and no horizontal "nominal" label; consider width and interval score only, and add the forecast-time-valid sets.
- **Table S4.** Clean longtable but contains no inventory source/citation, no mapping-extent flag, no mechanism and no positive-count caveat. Add source citations, flag the events with < 200 positive cells and use "n.a." instead of 0 for unknown stations.
- **SI** has no table of the regression results (offset_regression_summary), the lambda choices or the covariate definitions; S6/S7 headings are adjacent with Figure S2 floating after S7.
- **Typesetting.** main.log: 22 "Overfull" warnings (261.8 pt, all emitted "while \output is active", i.e. floats vs the Wiley class two-column output), many "Underfull \vbox (badness 10000)", and font-shape substitutions for stix2 bold. Title-page metadata still blank (Received/Revised/Accepted, volume 0, DOI 10.1002/0000). Page 1 header text is in the wrong template position for the NZJGG style. The abstract is OK at 288 words.

## 7. Prioritised action list

1. Rewrite the abstract, Conclusions 2-4 and Sections 5.1/5.4 to the 38-event numbers; remove "higher after wetter-than-normal months" as a finding; state that predictive-density gains are not significant and the point forecast did not improve (3.2-D/E/F, section 5).
2. Report the Kaikoura result under M0/M1/M4 (E/O 0.52/0.17/0.08; M4 misses) and rewrite the NZ implications accordingly; update the 9-event offset (-0.12) and "within 6%" statements.
3. Separate forecast-time-valid covariates (Mw, depth, mechanism, ShakeMap stations, climate with a forecast-time area) from inventory-defined covariates (type, mapping extent, p5 area, year) and re-run the conditional prior for the former (and for M1 with inventory type fixed to the planned mapping protocol). Explain legitimacy in the text.
4. Test the domain-definition confound: add log domain area and a forecast-time domain (shaking threshold) and show whether the Mw/size effect and the offset ordering survive (r = -0.53 with log area).
5. Fix the regression design: remove the duplicate `p5_missing`, extend the lambda grid beyond 100, report out-of-sample variance explained (M4 about 32%, M1 about 19%), the placebo (-0.025 vs +0.166), interval score, and multiplicity-adjusted p-values; drop "pre-specified" unless a dated record exists.
6. Report the subset results correctly: new-Pacific 12/16 (p = 0.065), non-Pacific 9/13, original nine 8/9 (p = 0.02); add an event-level bootstrap/permutation CI for the mean AUC difference and note that LOEO folds are dependent; add seed repeats for at least a subset.
7. Reconcile the 9-event and 38-event LOEO results for the original nine (6/9 vs 8/9; Aisen sign flip; Kaikoura E/O 1.06 vs 0.52) and decide which is primary.
8. Clean the covariates: ShakeMap station counts (separate unknown from zero), TerraClimate aggregation area and the dry-season ratio instability; add an indicator for unknown values and discuss events with fewer than 200 positive cells (Southern Italy 68, Livermore 105, San Miguel 134, Central Italy 176).
9. Cite the 29 new inventories (per-event DOI/citation in Table S4), TerraClimate, ComCat and Planetary Computer; update the Data Availability statement and the Pacific Rim definition (boxes).
10. Reorder the manuscript (Experiments 2/3/4 numbering, global extension placement), improve Figure 11 (labels, uncertainty, legend overlap) and add a regression-results table to the SI.
11. Fix typesetting: title-page metadata, overfull/underfull warnings where possible, adjust title to "Pacific Rim and beyond" or similar, harmonise "zero-shot".
