# Referee report, round 1
Manuscript: "Transferable coseismic landslide forecasting across the Pacific Rim with hierarchical physics-anchored boosting and selection-aware Bayesian updating" (NZ J. Geol. Geophys., special issue "AI and Remote Sensing for Environmental Geosciences").
Reviewer scope: PDF (22 pp.), LaTeX sources, src/, results/. Nothing was edited, committed or pushed.

## 1. Summary and recommendation

**Content.** Nine open coseismic landslide inventories (Kaikoura 2016, Northridge, Denali, Niigata-Chuetsu, Iwate-Miyagi, Aisen, Cinchona, Porgera, Palu) are combined with GLO-30 terrain, ShakeMap PGA/PGV and a generic Newmark displacement. Leave-one-event-out (LOEO) tests show that physics, logistic and boosted models rank cells moderately well in a new event (median AUC 0.75-0.81) but mispredict the landslide total by up to about 3 orders of magnitude (E/O 0.02-23). The authors propose HiPhyBoost with three components:
- (i) a Newmark-anchored, monotone boosted model with latent event offsets;
- (ii) an empirical-Bayes prior over offsets, giving zero-shot intervals for the landslide total;
- (iii) a quasi-likelihood, selection-corrected Bayesian update of the offset from k mapped tiles.

**Strengths.**
- The question is operationally important and the level-versus-shape diagnosis is a real contribution.
- The design is clear and the data are open.
- The AUC, E/O and bootstrap numbers I re-computed are almost entirely consistent with the result files (Section 2).
- The authors are candid about several negative results: Bayesian BSS, no SHAP stabilisation, the weak selection-term effect and the quasi-posterior's undercoverage.
- The Pacific Rim and Kaikoura framing fits the issue.

**Weaknesses.**
1. The headline gain of HiPhyBoost (+0.028 median AUC over PC-GBM) rests on 9 events. It is not significant at the event level (paired Wilcoxon on the 9 AUC differences, two-sided p = 0.16; vs GBM p = 0.07). The block bootstrap only quantifies within-event sampling noise.
2. The zero-shot "interval covers 7/9" is almost uninformative. Intervals span a factor of 59-389, i.e. 2-2.6 orders of magnitude. Point forecasts are no better than PC-GBM, and the Bayesian predictive mean is much worse (see M3).
3. The few-shot gain in point accuracy comes mainly from prior shrinkage, not from the "quasi-likelihood". The comparison is with unshrunk recalibration, so the baseline is unfair. Reported medians hide failures (Denali: 0% coverage for k<=20 and 26x error at k=1).
4. Several modelling choices are weakly justified: the selection term (an uncorrected plug-in approximation and a self-referential simulation protocol), the quasi-likelihood scale, and a single phi. There is mild target leakage in phi, sigma_u and the per-event oracle offsets.
5. Calling the offsets a "geoscientific quantity" is not demonstrated. They are confounded with domain definition, mapping completeness, ShakeMap bias and polygon-to-point reduction. "Consistent with independent evidence" is a post-hoc narrative for n=9 with no test.
6. The manuscript is long, with a 360-word abstract and a lot of internal redundancy. There are also typesetting faults.

**Recommendation: MAJOR REVISION.** The study is worth publishing in the special issue, but only after the inferential claims are re-scoped, fairer baselines and event-level statistics are added, leakage is removed, and the geoscientific interpretation of the offsets is tested rather than asserted.

## 2. Numerical consistency audit

I recomputed about 60 quoted numbers with pandas from results/*.csv, json and data/processed. Items 1-34 are correct. The mismatches and imprecisions are listed after the table.

| # | Quoted (location) | Source / recomputed | Status |
|---|---|---|---|
| 1 | Median transfer AUC PC-GBM 0.805, GBM 0.790, Physics 0.763, Physics-logit 0.764, LR 0.750 (Sec 4.1) | e1_loeo_metrics: 0.805/0.790/0.763/0.764/0.750 | OK |
| 2 | HiPhyBoost median AUC 0.833, AP lift 4.1; ablation 0.810 (abstract, 4.5) | e4: 0.833, 4.148, 0.810 | OK |
| 3 | Kaikoura AUC 0.917, AP lift 12.7 (abstract, 4.5) | e4: 0.917, 12.727 | OK |
| 4 | Worst-case AUC PC-GBM 0.730 (Palu), GBM 0.639, physics 0.559, LR 0.247/0.397 | e1 minima | OK |
| 5 | Local PC-GBM median AUC 0.838, AP lift 5.7; transferred 0.805/3.4 | 0.8376, 5.741; 3.438 | OK |
| 6 | Kaikoura transferred PC-GBM 0.887/AP lift 9.2; local 0.934/16.8 | same | OK |
| 7 | Gap Iwate 0.833 vs 0.834; Northridge 0.08 | 0.833/0.834; 0.079 | OK |
| 8 | PC-GBM vs LR: 7 events significant; Kaikoura +0.06; Cinchona 0.37, Aisen 0.51; Palu equivalent; Denali -0.024 | e1_auc_diff: 7 with CI>0; 0.060; 0.374; 0.512; -0.003 (CI spans 0); -0.024 | OK |
| 9 | PC-GBM vs GBM: Northridge +0.025, Niigata +0.014, Iwate +0.037, Aisen +0.119, Denali -0.045, Palu -0.059 | same. Niigata CI lower bound is 0.001 (marginal) | OK |
| 10 | HiPhy vs PC-GBM: Kaikoura +0.030 [0.021, 0.038], Northridge +0.019, Denali +0.042, Porgera +0.044, Palu +0.049, Aisen -0.025, Cinchona -0.013 | e4_auc_diff | OK |
| 11 | E/O PC-GBM 0.02 (Cinchona) to 23 (Denali); within factor 2 in 4 events | 0.021, 23.4; Kaik 1.51, Northridge 0.94, Aisen 1.11, Palu 1.89 | OK |
| 12 | Physics overpredicts in 8 events, x8 Kaikoura, up to 228; slopes 0.1-0.4; median E/O 8.3 | 8.28, 228.4, slopes 0.106-0.40 (Cinchona E/O 0.93 is the exception) | OK |
| 13 | PC-GBM calibration slope median 1.07, range 0.82-1.53 | 1.065, 0.819-1.530 | OK |
| 14 | BSS local 0.000-0.017; transferred -0.20 to +0.01 | local -0.0002 to 0.0172; -0.201 to 0.010 | OK |
| 15 | Offsets span 5.7 log-odds; Denali -3.15, Cinchona +2.59; tau 1.69; "300 times" | 5.745; -3.155; 2.590; 1.692 (ddof=1); e^5.745 = 313 | OK |
| 16 | Niigata +1.7, Northridge +0.7, Porgera -1.6; Kaikoura b=-0.12 | 1.741, 0.726, -1.622, -0.116 | OK |
| 17 | SD of offsets across LOEO folds 0.18-0.33 | 0.179-0.335 | OK |
| 18 | Zero-shot coverage 7/9; misses Cinchona (above), Denali (below) | e4 covered | OK |
| 19 | 90% interval spans a factor 60-390 | q95/q05 = 59-389 | OK |
| 20 | Zero-shot point forecast within x2 in 3 events (Iwate, Kaikoura, Palu); PC-GBM 4 | q50 E/O 1.12, 1.06, 1.24; Northridge 0.457 (just outside) | OK |
| 21 | Kaikoura HiPhy forecast within 6% of inventory; rho = 0.61 | 30 248/28 474 = 1.062; 0.6149 | OK |
| 22 | Spearman for Northridge, Iwate, Porgera, Palu 0.51-0.64 | 0.553, 0.642, 0.506, 0.588 | OK |
| 23 | Few-shot k=1: intercept recal. 2^1.31 = 2.5; cell-Bayes 2.2; quasi 0.68 -> 1.6; k=5-10 0.48-0.53 -> 1.4; PC-GBM 1.6 | recomputed 1.306, 1.142, 0.681, 0.526/0.484, 0.64-0.71 | OK |
| 24 | Cell-level coverage 4-26%; quasi 0.72-0.76 for k<=10, 0.70 at k=20, 0.54 vs 0.51 at k=40 | 0.042-0.263; 0.733/0.763/0.741/0.722; 0.696; 0.542/0.508 | OK |
| 25 | Quasi interval width 59 (k=1), 6.5 (k=5), 3.7 (k=10) | 59.0, 6.48, 3.72 | OK |
| 26 | Tile random effect: coverage 0.20 -> 0.74 at k=10; ~40% error reduction; bias x2.5-4.4 | 0.196 -> 0.741; 2.27 -> 1.37 (40%); 2^1.35 = 2.5, 2^2.15 = 4.4 | OK |
| 27 | AUC in few-shot 0.834-0.841 vs 0.805-0.816 | same | OK |
| 28 | Monotonicity violations GBM 5-43% (mean 26), LR 21-82% (mean 43) | 0.047-0.425 (0.257); 0.207-0.817 (0.426) | OK |
| 29 | SHAP stability: Spearman 0.42/0.31; Kendall W 0.48/0.38 | e3: 0.420/0.307; 0.485/0.384 | OK |
| 30 | LR sign change at PGV 3.6/9.0 cm/s; median PGV 3.95/10.05; 95.4% below training 1st percentile | diag_global_lr.txt | OK |
| 31 | Table 1 prevalence, positives (e.g. 28 474 Kaikoura) | results/ n/pos table | OK |
| 32 | Table 2 and Table 3 cell values | all rows spot-checked against e1/e4 | OK |
| 33 | Conclusion 4: point estimate off by 20-50x in the worst cases | 1/0.021 = 48; Denali 23 | OK |
| 34 | "Three orders of magnitude" | 0.021-23.4 = 1100x (for physics 0.93-228) | OK |

**Mismatches and imprecisions**
- **M-a, Sec 4.8 (Uncertainty) and Conclusion 5: phi "ranges from 7 (Denali) to 190 (Cinchona), median 55".** These are the target-event oracle dispersions (phi_target_oracle in e7_phi.json: 7.35-193.4, median 55.4), not the values used in the model. The phi actually used is the median over the other eight events: 46.6, 59.0 or 67.8 only, because only three distinct values arise. The text therefore describes a quantity the algorithm never uses. "Roughly one fiftieth" of information holds for the oracle median, not for what is applied. Also 193 -> "190" should read "about 190".
- **M-b, Sec 5.1: "median PGA of only 0.19 g at the landslide cells".** Table 1 gives the *mean* PGA at landslides (0.19 g). The ShakeMap maximum 0.27 g is correct.
- **M-c, Table 3, bold typesetting.** HiPhyBoost is bold in every row, including Niigata (0.784 vs PC-GBM 0.788), Aisen and Cinchona, where it is significantly *worse*. The caption says bold marks nothing in particular, but the asterisk and bold imply "best". Bold the best model per row.
- **M-d, Sec 3.5/4.5 vs results.** Methods says "for HiPhyBoost we report both the plug-in prediction and the zero-shot Bayesian prediction". Only the plug-in/median is reported in the text. The Bayesian zero-shot predictive mean has median E/O = 3.73 (Kaikoura 3.73, Denali 67, Porgera 19, Palu 4.9, Iwate 4.3; e4_hiphy_loeo.csv), worse than PC-GBM (1.50). It is also not within 2x in 4/9 events (BSS median -0.031). This must be disclosed.
- **M-e, Abstract.** "26-43% of cells" is the *mean over events* (GBM, LR), while per-event ranges are 5-43% and 21-82% (Discussion). State "on average 26% (GBM) and 43% (global LR) of held-out cells".
- **M-f, Table 3 / Fig. 3 AP lift.** The text says HiPhyBoost (4.1) beats "3.9 for GBM". It does not mention that Global LR (3.94) and GBM (3.94) have higher median AP lift than PC-GBM (3.44), and that LR beats PC-GBM at Kaikoura (9.9 vs 9.2). "PC-GBM is best" holds only for AUC.
- **M-g, Sec 5.4 (NZ).** The text says Kaikoura's offset is pushed *upward* by complete mapping, yet the fitted b = -0.12 (the oracle value on the E7 scale is -0.06). Either way it is not above the mean, so the argument is not supported by the numbers.
- **M-h, results_fewshot.** The k=40 values are over 8 events (Niigata is skipped by `k >= len(ub)//2`), and k=20 over fewer where tiles are scarce. This is not stated in Table 3 or Fig. 5. Median AUC and E/O at k=40 are not comparable with k<=20.
- **M-i, Sec 4.6.** The point-forecast "within factor two" comparison counts q50 for HiPhyBoost against the PC-GBM mean prediction (E/O). These are different functionals.

## 3. Methodological concerns

### M1. LOEO design and event-level inference (MAJOR)
Everything is evaluated on 9 events, but uncertainty is quantified only with a spatial block bootstrap (200 resamples, fixed fitted models). That ignores model-training variance and between-event heterogeneity, which dominate for claims of the type "method A is better in general".
- Event-level paired AUC differences, HiPhyBoost vs PC-GBM: mean +0.016, median +0.019, 6 of 9 positive, Wilcoxon p = 0.16. The text's "significantly better in five of nine events" is event-wise within-event significance.
- HiPhyBoost is *worse* at 4 events (Niigata, Aisen, Cinchona and, by AP lift, Iwate), i.e. in the low-shaking events where transfer is hardest.
- 200 bootstrap resamples make 95% interval endpoints noisy. Use >=1000 and report whether blocks (5 km) exceed the autocorrelation range.
- The hyperparameters of HiPhyBoost differ from "the same" ones as PC-GBM: T = 3 outer rounds with 150/150/400 trees (700 trees) vs 400. The model also gets an extra logistic anchor and 8 free event parameters. No seeds, repeated fits or tuning/robustness checks are given.
- *Fix:* report event-level paired tests and a hierarchical or leave-one-event-out bootstrap (resample events); repeat with several seeds; equalise tree budgets; do a sensitivity analysis on the PC-GBM/HiPhyBoost hyperparameters.

### M2. Data leakage in the few-shot calibration (phi, sigma_u, oracle offsets) (MAJOR, though modest in size)
- In `13_fewshot_quasi.py`, phi for target T is the median over the other 8 events of the Pearson dispersion computed from `hiphy_loeo_scores.parquet`. The score of event e comes from the LOEO model that excluded e but was trained on **T** (and the other 7). So T's inventory influences phi_e through g, and phi_e is computed with e's own *full-inventory oracle offset* (`brentq` on all cells). Both are illegitimate for a genuinely new event.
- Likewise, sigma_u and the tile-RE prior (mu, tau) in `12_fewshot_overdispersed.py` use `bcond` offsets of the other events estimated from LOEO scores obtained with models that saw the target.
- The Methods text ("estimated from the out-of-sample scores of the training events") hides this. It is out-of-sample for e, not for T.
- Because phi takes only three distinct values (46.6, 59.0, 67.8) the leakage may be numerically small, but it must be removed or bounded.
- The simulated-mapping protocol (random tiles among *positive* tiles, uniform) is exactly the one the selection term assumes. The "selection-aware" gain is therefore never tested under a mis-specified selection mechanism (e.g. selection proportional to visible landslide density or accessibility), which is the realistic case. The selection term is shown to be ~null (0.54 vs 0.51 coverage at k=40 only), so the "selection-aware" label in the title is not supported by the evidence.
- *Fix:* nested design: for target T, refit the model excluding T for each source event e (i.e. 8 x 7 nested LOEO), or use phi from a fixed pilot set; report phi from a leave-two-out scheme; report sensitivity of coverage to phi (0.5x-2x); test selection under biased tile choice (e.g. probability proportional to landslide count) and under missed detections.

### M3. Zero-shot forecast is weakly informative and the Bayesian mean is not calibrated (MAJOR)
- The prior is N(mu = 0, tau) where mu is exactly zero by construction (offsets are centred) and tau is the SD of 8 *estimated* offsets (1.29-1.88). The interval ignores uncertainty in tau and mu. A t-type predictive (n=8; t_{0.95,7} = 1.895 instead of 1.645, plus sqrt(1+1/8)) would be about 22% wider on the log scale.
- 7/9 = 0.78 coverage at nominal 0.90 with n = 9 cannot be distinguished from 0.9 or from 0.6. With intervals spanning 2 to 2.6 orders of magnitude, coverage is nearly trivial. Report interval *sharpness* against the trivial benchmark ("the landslide total lies within a 1.7-sigma lognormal about the median across past events") and a proper score (CRPS or interval score), and compare with a naive prior on the raw log landslide count of past events scaled by shaken area.
- The zero-shot Bayesian predictive mean has median E/O = 3.7 (M-d). Averaging sigma(m + b) over N(0, 1.7^2) inflates expected totals (Jensen on a convex-then-concave link). The manuscript's headline point forecast is the median of the offset distribution, a different object from "p^(0)" in Eq. (3). Choose and define one.
- Neither model "can predict the base rate", as the text says, yet the abstract frames "zero-shot predictive intervals" as an achievement. Be explicit that the intervals are a prior-ignorance statement, not forecast skill.
- *Fix:* add a covariate model for b_e (even a 1-parameter one: Mw, rainfall, inventory resolution), or an event-level random-effects meta-regression, and test with LOEO whether it narrows the interval. Without this, component (ii) is a descriptive statement about 9 numbers.

### M4. Quasi-likelihood justification and phi (MAJOR)
- Dividing the log-likelihood by phi is a *power posterior / quasi-Bayes* device. It inflates variance by phi, but the posterior is no longer a coherent Bayesian update, and the credible-interval interpretation depends on a correct phi. Coverage is only 0.72-0.76 at nominal 0.90 and 0.54 at k=40; this should be described as "improved but still overconfident".
- phi is a Pearson dispersion for **3 x 3 km tiles at fixed tile size**. Clustering-induced overdispersion scales with the number of cells per tile, so phi is tile-size dependent and not transferable to the unmapped remainder or to other tile sizes. It is also dominated by a few high-count tiles, and varies 7-193 between events (26-fold), while the model uses ~47-68. The text acknowledges this but does not quantify the sensitivity.
- The posterior is evaluated for a 1-D offset on a grid, which is good. But tile counts are heavy-tailed (most tiles zero, a few hundreds). A negative-binomial or Poisson-lognormal tile model with a *tile-count* likelihood (instead of per-cell Bernoulli divided by phi) would be a natural and defensible alternative. The Gaussian tile-RE variant is a weak competitor; its failure does not prove that the quasi form is right.
- At Denali the quasi interval never covers for k<=20 (coverage 0.00; E/O 26 -> 3.9 across k; the oracle offset -3.55 lies >2 tau from the prior). At Cinchona coverage at k=1 is 0.2. So the "median" accuracy and "mean" coverage hide systematic failure in the extreme events, exactly those where forecasts are needed.
- Point-error gains at small k come from shrinking toward the prior (the quasi posterior inflates variance, so the mean moves to the prior). Fair comparators are: (a) PC-GBM intercept recalibration with a Gaussian prior on the intercept (empirical Bayes with the same mu, tau); (b) the cell-level HiPhyBoost posterior with a tempered prior. Without this, "keeps the best point accuracy of all methods" is a statement about shrinkage vs. none.
- *Fix:* add (a), (b); report per-event results (a table) not just medians across events; report sensitivity to phi; consider estimating phi on the target's own mapped tiles (with k>=5) via a hierarchical model.

### M5. Selection-term approximation (MAJOR for the "selection-aware" claim; otherwise minor)
- The exact selection likelihood of the observed tile set is E[1/C(N+,k) | y_C]. The authors replace the random N+ by its expectation inside C(.,.), a Jensen-gap approximation. N+ is a sum over strongly dependent tiles (clustering), so Var(N+) is large and the plug-in is biased. In addition, q_t(b) assumes independent cells within a tile (the very assumption rejected for the likelihood), so N+ is overestimated.
- Dividing the selection term by phi (as is done in Eq. 5) has no probabilistic justification; the authors concede it dilutes the term. Either the term is effectively off (as results show) or it should enter undivided.
- The selection model is "uniform among positive tiles". Real analysts pick high-density and visible tiles. The term should be tested against a size-biased selection.
- *Fix:* either demonstrate a benefit under a misspecified (size-biased) selection protocol, or drop selection from the title and abstract and relegate it to a supplement. The title currently advertises a component that, by the authors' own Results, has almost no effect.

### M6. Are offsets "geoscientific"? (MAJOR)
- The offsets absorb everything not in the 11 predictors: antecedent wetness, lithology and material strength, ShakeMap bias, inventory completeness and minimum mapped size, polygon-to-point reduction, DEM post-dating, and **domain definition**. For 6 of 9 events the domain is the convex hull of the inventory buffered by 2 km (Denali 15 748 km2 hull with 1579 landslides; Porgera 24 306 km2), so prevalence and b_e depend on where the authors draw the boundary. The domain-size/prevalence correlation is strong in Table 1.
- With n = 9, the "consistency with independent evidence" (Sec 5.1) is a narrative fit: Denali low (explained by rupture characteristics in Gorum et al. 2014/Jibson 2004), Niigata high (rainfall), Cinchona high (weak volcanic soils/ShakeMap), Porgera low (forest/cloud). Each story is plausible, none is tested; opposite outcomes could be explained equally easily. Inventory-completeness arguments are used both ways (Porgera low; Kaikoura "more complete" but b ~ 0).
- Statements like "rainfall shortly before an earthquake can raise the offset by more than one log-odds unit (Sassa 2005)" (Sec 5.4) are not supported: the citation does not give an offset and there is no counterfactual in the model.
- Offset stability across LOEO folds (SD 0.18-0.33) is largely mechanical: each fold's eight-event fits share seven events, so it shows numerical stability, not "a robust property of the data". It says nothing about identifiability versus artefacts.
- *Fix:* (i) test for domain-definition sensitivity (re-estimate b_e on a common-resolution buffer, e.g. 5 km around the 90% shaking contour, or restrict to MMI>=VI cells); (ii) regress b_e on event covariates available from ComCat/rainfall reanalysis (e.g. 7-30 d antecedent rainfall, Mw, depth, mapping resolution, mean ShakeMap uncertainty) and report the rank correlations and leave-one-out predictive error; (iii) compare ShakeMap-based and independent shaking (e.g. different ShakeMap versions, or PGA vs MMI); (iv) call b_e an "event-level residual propensity (including observation effects)" until (i)-(iii) succeed. Remove "geoscientific quantity" from the Conclusions or qualify it.

### M7. Fairness of baselines and ablations (MAJOR/minor)
- The global-style logistic regression has 4 predictors (log PGV, slope, interaction, relief), no lithology/CTI/land cover/wetness, no regularisation, and is called "global-style" and "mirrors the USGS model". Its inversion at Aisen/Cinchona is a property of this specification (an unconstrained interaction extrapolated to PGV outside the training range), not of operational USGS models. The headline failure ("LR AUC 0.247") should be labelled as such.
- GBM and PC-GBM were never tuned (stated). That is defensible as a fixed protocol, but it is not "fair" in the sense claimed because the proposed method gains extra capacity; add at least a small hyperparameter sensitivity.
- Missing baselines that isolate the contribution of the hierarchical structure: (1) PC-GBM with event-balanced weights plus a random-intercept logistic mixed model on the Newmark feature only; (2) PC-GBM trained with event dummies, then predicted with the mean dummy; (3) the "no offsets" ablation with the same T=3 rounds, tree budget and anchor (the present ablation already behaves like PC-GBM, which is good, but its relation to PC-GBM should be stated: 0.810 vs 0.805 vs 0.833).
- Few-shot baselines are unshrunk recalibration (see M4). Add a shrinkage-regularised (Bayesian) recalibration baseline.
- The Newmark-Weibull "Physics" baseline is calibrated on Northridge, a single event that is itself in the test set (the curve was *fitted on* the Northridge inventory, so its Northridge LOEO row is not out-of-sample; footnote it). It is described as "zero-parameter".

### M8. Sampling and weighting (minor)
- 2.5 x 10^5 uniformly sampled negatives per event with inverse-fraction weights is fine for AUC/E/O, but weighted AP lift and BSS at prevalence 0.01-1% have high variance for the rare-event tails (Denali: 1579 positives). No interval on AP lift is given.
- Event-balanced training weights mean Aisen (512 positives) and Kaikoura (28 474) count equally. This is a design choice that should be tested against pooled weights, since it interacts with the offset estimate (offset solved per event with weights).
- One polygon -> one point -> 30 m cell: landslide size is discarded, so the "number of landslide cells" is a count of sources, not area or volume. E/O comparisons across inventories with different polygon splitting/minimum size are not like-for-like. Provide sensitivity to a 100 m positive definition.

### M9. Statistical significance of attribution results (minor)
- Mean pairwise Spearman correlations of SHAP importance (0.42/0.31, SD 0.26/0.31) are computed on 11 *collinear* predictors (PGA, PGV, D_N, slope, slope_max5, relief). TreeSHAP splits credit among correlated features arbitrarily, so low rank agreement is expected even if the *same physics* operates. The conclusion "attributions do not transfer" is therefore confounded. Group correlated predictors (shaking, slope, relief) and compare group importance, or use an interventional/grouped SHAP, and give a null reference (permutation of event labels).
- The constrained-model statement that PGA thresholds differ by "almost an order of magnitude" is read from binned SHAP medians (Fig. 12b), without uncertainty.
- The perturbation test (+10% PGA/PGV) on piecewise-constant trees detects any step crossing in either direction; compare against the sampling noise of a retrained unconstrained model, and note that decreasing responses may reflect genuine data signals (e.g. site effects), not "implausibility".

### M10. Zero-shot interval, "total" definition and spatial skill (minor)
- The "total" is the sum over all domain cells, which includes low-shaking cells that dominate area. Report the total over cells above a shaking threshold.
- The 2 km Spearman rho = 0.51-0.64 is given without a reference (e.g. PGA-only or Newmark-only maps) and with no uncertainty. Spearman on 2 km cells with many zeros is dominated by the zero-inflation. Compare to the same statistic for PC-GBM and for the local model.

## 4. Claims that overreach or lack support

1. Abstract/Conclusion 3: "learned event offsets ... consistent with independent evidence on antecedent wetness, weak soils, shaking bias and inventory completeness". Not tested (M6). Replace with "plausible in light of ...".
2. Conclusion 3: "the event offsets are a geoscientific quantity". Overreach (M6).
3. Conclusion 2: "Separating the shared response from latent event offsets improves transfer", "significantly better in five of nine events". Within-event bootstrap only; at event level p = 0.16 and it is worse in 4 events (M1).
4. Abstract: "gives intervals that are far better calibrated than those of a naive update". True relative to a strawman (cell-level, 4-26% coverage). Absolute coverage is 0.72-0.76 (0.54 at k=40) and 0% at Denali.
5. Abstract/Conclusion 5: "keeps the best point accuracy of all methods" (median across events; at k=1 Denali is off by 26x and Porgera 5.6x). Report worst-event errors.
6. Abstract: "median factor of 1.4-1.6 after one to ten tiles" is OK numerically but presented without the heterogeneity. Add the range across events.
7. Conclusion 6 and Sec 5.3: monotone constraints "prevent physically implausible responses". For PC-GBM this holds by construction for the 3 shaking terms, not for the whole model, and the premise that a probability must be monotone in ShakeMap PGA is itself disputable (Palu flowslides, Denali), as the authors note. "A model whose forecast falls when shaking is revised upward is hard to defend" is a good argument but is not a result.
8. Sec 4.3: "the miscalibration ... is mostly a shift in level, not in shape" because calibration slopes are near 1 (median 1.07). A slope near 1 *with E/O far from 1* identifies a level shift only if slopes are estimated after intercept re-fitting and per event; at Denali the slope is 1.53, Porgera 1.40, and Physics-logit 1.5 (median). The statement should be restricted to the PC-GBM. A direct test is to compare calibration slope after fitting an event-specific intercept (cal_int vs cal_slope).
9. Sec 5.4: "Kaikoura ... transfers almost without bias ... encouraging for Alpine Fault or Hikurangi". Kaikoura is a favourable case for the model (b ~ 0, high-quality ShakeMap, mapped at 0.3 m). Generalising to Alpine Fault (Alpine Fault ruptures: long surface-rupture on a plate boundary with different geology: schist/greywacke; high rainfall) needs explicit argument; zero-shot interval spans 270x at Kaikoura.
10. Sec 5.4: "Section 4.7 shows that a handful of tiles is enough to reduce the uncertainty substantially". Interval narrowing (59x -> 3.7x) is a property of the posterior under an assumed phi, with coverage still below nominal. A factor-3.7 interval after 10 tiles is not necessarily "enough" for decisions.
11. Intro: "Spatially autocorrelated samples inflate random CV scores, by 6-18% in a recent regional comparison (kumar2025crossvalidation)". This matches the Crossref abstract in recent_refs_notes.md, but check that the number is for the *same* metric/task.
12. Citations whose notes mark them as **title-only / abstract not available** are cited for content beyond the title:
   - `zhang2024nearrealtime` (Intro: "trains near-real-time models on global inventories") is acceptable; the note says title only, so do not describe its results or comparison to it.
   - `ma2026crossevent` (Intro: "assesses cross-event transfer between the Chi-Chi and Hualien earthquakes") is title-level only. Do not describe methodology/performance.
   - `wu2026hierarchical` (Methods: "related to hierarchical Bayesian susceptibility modelling for data-sparse regions") is title-only; "related to" is the safe maximum.
   - `loi2024noto`, `zhang2025noto`, `chang2024hualien`, `chen2025hualien` (Intro: "struck steep, populated Pacific Rim terrain"; Noto/Hualien) are title-only. The sentence says more (steep, populated terrain) than the titles support; cite only for "landslides were reported". `zhang2025noto` is again cited as a candidate zero-shot test event, with no data check.
   - `jiang2026frequency` (Limitations: "effective frequency content relevant to slope response is not represented by peak values") is title-only; the specific claim may not reflect the paper's conclusion.
   - `nasios2026multimodal`, `endo2026realtime` (abstract available, ok), `xu2026rapid` (F1 etc. from the OpenAlex abstract, ok).
   - Rainfall-triggered / non-coseismic references used as support for coseismic transfer: `su2025multisource`, `wang2026crossregional`, `wang2026regionsimilarity` (Intro "domain adaptation ... shown to improve transfer"), `wang2024shap` (not coseismic), `massey2025gabrielle`, `dreyer2026climate`, `howard2025muriwai` (rainfall). The Intro says "for rainfall-related susceptibility", which is correct, but Sec 5.4's use of `massey2025gabrielle, dreyer2026climate` to argue that an Alpine Fault earthquake after a wet period "would plausibly sit on the positive side" is speculation; none address coseismic offsets.
   - `chen2026ebm` (single-event random split AUC 0.976) is cited as a recent "interpretable boosting" coseismic study; fine, but note that it is a random-split result.
   - Date integrity: several 2026 items are online-first without volume/pages (`yilmaz2026kahramanmaras`, `ma2026crossevent`, `wu2026hierarchical`, `xu2026rapid`); the notes file already flags a re-check. `zhang2025noto` and `liu2025newmarkcnn` appear with issue year 2025 but are in 2026 volumes.
   - `sassa2005` is cited for "rainfall shortly before an earthquake can raise the offset by more than one log-odds unit": the paper reports rainfall and the earthquake, not an offset. The quantity is the authors' own.
   - `gorum2014denali` and `jibson2004` are cited for "explanations involve rupture and site characteristics that a PGA/PGV field does not capture". The sentence attributes content to them with "discussed in those studies" without saying what; check and state exactly (Gorum et al. discuss the sparse landsliding in terms of seismic and geological factors; Jibson et al. 2004 argue low-frequency/long-period shaking).
13. Sec 4.8 "A Gaussian random effect cannot represent the real distribution of landslides over tiles" is plausible but not tested against a heavy-tailed tile effect (see M4).

## 5. Presentation

**Structure and length.**
- The document is 22 pages and about 16 000 words including references, supplement and tables, which is long for a journal paper. There is a lot of internal redundancy: the same results appear in the Abstract, Sec 4, Sec 5.1, Sec 5.4 and the six-point Conclusions; Fig. 3, Fig. 4 and Table 2 show the same E/O and slope information; Table S2 (full metrics) repeats Table 2.
- Suggested cuts (target about 12-14 pages): (a) merge Sec 4.1-4.3 into one half-length section and move Fig. 4 (reliability) and Table 2 to the supplement; (b) drop Fig. 5 panel (a) (AUC is unchanged by updating, by construction, and the shading obscures the lines), and report one table for the few-shot results; (c) condense the algorithm text (Algorithm 1 + Eq. 3-5) and the Gaussian tile-RE variant to a supplement; (d) cut the Newmark and baseline sections to what is novel; (e) shorten the NZ and limitations paragraphs by about a third; (f) merge or drop the SHAP section into a short paragraph (it is "negative" evidence about a method that is not the paper's subject) or move it to a supplement.
- The Introduction is overloaded with references (about 80) and the "four gaps" list partly duplicates the contributions list. The recent_refs insertion is visible as "citation stuffing" (e.g. Noto/Hualien sentence).

**Abstract.** About 340-360 words against a norm of 250-300. It lists all three components with numbers and the full NZ recommendation. Cut the event list (move to Methods), shorten the three components, and keep 4-5 headline numbers. State uncertainty (n = 9; 7/9 coverage).

**Terminology.**
- "PC-GBM" (physics-constrained GBM) vs "monotone-constrained boosted model" vs "constrained boosting" vs "physics-anchored" are used interchangeably; the "No offsets" ablation is "PC-GBM-like" in code. Define PC-GBM once, and refer to the HiPhyBoost parts consistently as (i)/(ii)/(iii) or "offset model / EB prior / quasi-posterior".
- Component names in Table 3 ("quasi-Bayes", "cell-level Bayes", "tile random effect") do not match the names in text ("cell-level posterior", "quasi-posterior", "Gaussian tile-random-effect formulation") or in the files ("naive", "overdispersion"). Harmonise.
- "Zero-shot" is used for both the median-offset point forecast and the integrated predictive mean (M-d).
- "Offset" (b_e), "event propensity" and "base rate" are used loosely; define b_e as a log-odds *intercept shift on the anchored scale*, not a base rate.
- Eq. (3) writes tau = SD; Algorithm 1 uses sd(b_e) with ddof unspecified (code: ddof=1).

**Figures and tables.**
- Page 1: a crossed-out placeholder box (template logo area) overlaps the title text. Remove it.
- Fig. 8b (offsets): the legend overlaps the Denali interval; the star and dot positions are hard to distinguish for small intervals. Add numeric E/O labels or move the legend outside.
- Fig. 5: panel (a) is uninformative and shading overlaps; panel (c) mixes a nine-event mean with the quasi-posterior failures; add per-event thin lines or an event-level dot plot for Denali/Cinchona.
- Fig. 3 (five models x nine events, three panels) is dense; the point of the local-model bars is lost.
- Tables 2-4 use `\resizebox{\textwidth}{!}` (tiny font in two-column layout; fonts differ between tables). Use table* with natural widths or reduce columns.
- Table 3 caption does not say what the asterisk means beyond "95% bootstrap interval excludes zero", and bold implies best model (M-c). Table 4 states "median over events of per-event medians" and "coverage ... " but not that k=40 excludes Niigata (M-h).
- Table 1 is placed before Fig. 1 and Fig. 2 in a "Results" section though it describes data; move Table 1, Fig. 1 and Fig. 2 to Sec 2 (Data).
- Figures 1-2 and Tables 1-4 are all `[t]` floats in `table*`/`figure*` (full width), which pushes them to the next page; several figures appear 1-2 pages away from their first mention (Fig. 0 on p.5 but called in p.4; Figs. 6-9 pages 9-12). The reading order (Fig. 4 reliability diagram appears after the text on Brier score) is awkward.
- Fig. 4 (log-log reliability for LR vs PC-GBM only) uses 12 quantile bins with a log scale: show confidence bands.
- The reference list is formatted in two columns across ~4 pages; fine. Supplementary Table S2 is rendered across a full page at tiny font; consider a CSV supplement.

**LaTeX / typesetting problems seen in main.log and the PDF.**
- 23 overfull/underfull warnings, including several "Overfull \hbox (261.8pt too wide) while \output is active" and "Overfull \vbox (49.8pt too high)" around the Introduction output (floats too wide for the column or an `\includegraphics[width=\textwidth]` placed in a one-column float), plus ~40 underfull hboxes in the long paragraph at lines 45-46 of one section (probably the Limitations paragraph, justified text with long citations).
- `fixltx2e` and `amssymb` redundancy warnings, `multicol` with `twocolumn`. Harmless but untidy.
- The title block says "Received: Revised: Accepted:" (blank) and `\articledoi{10.1002/0000}`, `\volume{0}` placeholders remain.
- Corresponding author and "presentaddress" both list different people; confirm.
- Subsection-numbering oddities: "3.1.1 | (i)", "3.1.2 | (ii)", "3.1.3 | (iii)" appear as run-in headings, but the PDF shows them as subsubsections inside 3.3 — check that numbering matches cross-references ("Sections 4.4-4.7" in the Results opener; "Supplementary Figure S1" labels are fine).
- Algorithm 1 is split across the page break near the middle of p.6 with the one-column text; use `algorithm*` placement at top of a page.

## 6. Fit to the special issue; strengthening the geoscience and the Pacific Rim / NZ relevance

**Fit.** The paper addresses the special-issue theme (open Earth-observation inputs, physics-informed and explainable ML, Pacific Rim, Kaikoura). Its best geoscientific message is that between-event variation in landslide propensity, not within-event ranking, limits transfer. That is under-exploited; the paper currently spends more space on the Bayesian machinery than on the geoscience.

**Concrete suggestions.**
1. Explain offsets with data (M6): antecedent rainfall (ERA5-Land 7-30 d), lithology classes (GLiM), Vs30 or topographic proxies, depth/Mw/mechanism, mapping resolution and minimum polygon size, ShakeMap station density/uncertainty. Even with n = 9, a small table of rank correlations plus leave-one-out prediction of b_e would convert narrative into evidence and give a physically motivated prior (a regression prior in component (ii)).
2. Add Newmark-physics depth: use event-specific or lithology-based strength (the authors note this), a wetness term for the offset (pore-pressure ratio) so that b_e is partly interpretable as a change in effective strength, and test whether b_e maps onto an effective cohesion/friction shift via the anchor beta. Report the anchor coefficients (alpha, beta) and their stability across folds, and whether g is small relative to phi (what fraction of the signal is "physics" vs "residual").
3. Show response curves from HiPhyBoost (partial dependence of g and the anchor) against the Newmark model for slope, PGA and PGV to demonstrate "physics-informed" behaviour; currently the SHAP section is about event-specific models, not HiPhyBoost.
4. NZ relevance: (a) add a leave-NZ-out sensitivity (Kaikoura is the only NZ event; the claim "no NZ training data" is a statement about one event); (b) quantify what a Kaikoura-informed prior would do for an Alpine Fault scenario by simulating the zero-shot forecast on an Alpine Fault ShakeMap scenario and reporting the interval on the number of landslide cells or landslide area per km2 with the DEM and the Southern Alps ShakeMap (this would directly deliver the "scenario forecast" the abstract promises); (c) discuss Hikurangi (offshore subduction, lower PGA onshore, near Aisen/Cinchona regime, where HiPhyBoost lost) honestly: the proposed method is *worse* in precisely the low-shaking regime that may characterise Hikurangi events; (d) relate to NZ-specific tools (GNS Science landslide susceptibility/QRA work, NZ Landslide Database, the 2016 Kaikoura mapping, Cyclone Gabrielle inventory) and to how the tile mapping would be organised (GeoNet, GNS, satellites).
5. Provide an operational description: latency (ShakeMap availability, DEM processing), what an emergency manager would see (map of expected density with interval), and how the "tile" workflow would use SAR/optical detections with false positives and missed landslides (an explicit detection model in the likelihood).
6. Add Pacific-Rim events with recent inventories (Noto 2024, Hualien 2024, Kahramanmaras is not Pacific Rim) as a *true holdout* test set, evaluated after freezing the model. A prospective-style holdout would be far more convincing than LOEO on the same 9.

## 7. Prioritised action list

1. Add event-level statistical inference: paired tests or event-resampling intervals for all headline differences (HiPhyBoost vs PC-GBM/GBM), repeat training seeds, equalise tree budgets; state plainly that the AUC gain is small (+0.016 mean / +0.028 median), non-significant across events (p ~ 0.16), and negative in 4 events (M1).
2. Remove leakage in phi, sigma_u and the oracle-offset-based quantities (nested LOEO or independent pilot set) and report the sensitivity of coverage and error to phi (M2, M4).
3. Add fair few-shot baselines: shrinkage (empirical-Bayes) intercept recalibration of PC-GBM and a tempered cell-level posterior, plus a count-based tile model (negative binomial / Poisson-lognormal); report per-event results (not only medians), especially Denali (0% coverage, 26x error at k=1), Cinchona and Porgera (M4, Sec 4).
4. Decide the status of the selection term: either show a benefit under a mis-specified, size-biased selection protocol or remove it from the title and abstract and put it in a supplement; fix or justify the Jensen-gap approximation and the division by phi (M5).
5. Define and consistently report the zero-shot forecast: disclose the Bayesian predictive mean (median E/O 3.7) and the plug-in/median version; add a proper interval score or sharpness benchmark against a naive prior and a t-based interval accounting for estimation of mu, tau; reword "covers 7 of 9" (M3, M-d).
6. Test the offset interpretation: domain-definition sensitivity, event-level covariate regression with LOO, check Kaikoura/Porgera completeness arguments; reword "geoscientific quantity" and "consistent with independent evidence" unless supported (M6, Sec 4 claims 1-2, 12).
7. Fix the numerical/text mismatches in Section 2: phi description (7-193 oracle vs 47-68 used), PGA mean/median, Table 3 bolding, abstract "26-43%" wording, AP-lift statement, k=40 sample size and Kaikoura "pushes upward" statement.
8. Strengthen baselines (global LR with regularisation and, if possible, lithology/wetness; tuned or sensitivity-tested GBM; mixed-model Newmark logit baseline) and footnote the in-sample Northridge calibration of the Jibson curve (M7).
9. Reconsider the SHAP analysis: group correlated predictors, add a null reference, or reduce to a short supplement (M9).
10. Cut length to about 12-14 pages and the abstract to <= 300 words: merge Results 4.1-4.3, move Fig. 4/Table 2/Table S2 and the Gaussian tile-RE variant to the supplement, drop Fig. 5a, remove duplicated statements between Abstract/Discussion/Conclusions (Section 5).
11. Repair typesetting: title-page placeholder box over the title, overfull boxes, tiny `\resizebox` tables, float placement, placeholder metadata (volume, DOI, dates), harmonise terminology (PC-GBM, component and variant names).
12. Audit the citations flagged as title-only (`loi2024noto`, `zhang2025noto`, `chang2024hualien`, `chen2025hualien`, `zhang2024nearrealtime`, `ma2026crossevent`, `wu2026hierarchical`, `jiang2026frequency`, `nasios2026multimodal`) and the rainfall-only references; restrict claims to what titles/abstracts support and complete 2026 online-first metadata.
13. Strengthen NZ relevance: an Alpine Fault/Hikurangi scenario demonstration with the interval, a leave-NZ-out discussion, honest treatment of the low-shaking regime where HiPhyBoost does worse, and a frozen-model test on a post-2023 Pacific Rim event (Noto 2024 or Hualien 2024).
