**To:** ranawaqaraslam@whu.edu.cn
**Cc:** minh.dinh@maeducation.com; datla@uef.edu.vn
**Subject:** Expression of Interest – NZJGG Special Issue "AI and Remote Sensing for Environmental Geosciences"

Dear Dr. Aslam,

On behalf of my co-authors, I would like to submit an Expression of Interest for the Special Issue "AI and Remote Sensing for Environmental Geosciences: Data-Driven Approaches to Geology, Environmental Processes and Natural Hazards" in the *New Zealand Journal of Geology and Geophysics*.

**Preliminary title**
Transferable coseismic landslide forecasting across the Pacific Rim and worldwide with hierarchical physics-anchored boosting and overdispersion-aware Bayesian updating

**Indicative authors and affiliations**
- Quang-Vinh Dang, British University Vietnam, Hung Yen, Vietnam (submitting author)
- Dat Le, Department of Science, Technology, and International Projects, Ho Chi Minh City University of Economics and Finance, Ho Chi Minh City, Vietnam
- Minh Ngoc Dinh, Data Science and Information Technology, Millennia University, Ho Chi Minh City, Vietnam (corresponding author)

**Scope, methods and expected geoscientific contribution**
In the first hours after a large earthquake, landslide forecasts have to come from models trained on past events, because the new event has not yet been mapped. We test how well such models transfer between earthquakes, using only rapidly available Earth-observation inputs: Copernicus GLO-30 terrain (TanDEM-X), USGS ShakeMap ground motion and Newmark sliding-block displacement. The test covers nine Pacific Rim events with open inventories, including the 2016 Kaikōura earthquake, and then 38 earthquakes worldwide.

Physics-based, logistic and machine-learning models rank hillslopes reasonably well in an unseen earthquake. However, they mispredict its total amount of landsliding by up to three orders of magnitude, mostly as a shift in level rather than in shape. We therefore propose HiPhyBoost, which has three components:
1. A Newmark-anchored, monotone-constrained boosted model with latent event offsets.
2. An empirical-Bayes prior over these offsets that gives zero-shot predictive intervals for the landslide total.
3. A quasi-likelihood Bayesian update from a few rapidly mapped image tiles that accounts for the spatial clustering of landslides.

Main results:
- Across 38 earthquakes, HiPhyBoost transferred better than a physics-constrained baseline in 29 events, including 21 of 29 events not used in its development. It reached AUC 0.92 for Kaikōura without New Zealand training data.
- With one to ten mapped tiles, it predicted the remaining landslide total within a median factor of about 1.5, with much better calibrated uncertainty than standard recalibration.

The geoscientific contribution is the interpretation of the event offsets as each earthquake's landslide propensity relative to generic slope mechanics. We examine how they relate to antecedent wetness, ShakeMap quality, faulting style and inventory practice. The manuscript closes with implications for scenario forecasting of Alpine Fault and Hikurangi earthquakes and for rapid-mapping strategies in Aotearoa New Zealand.

A complete draft manuscript is available and can be shared on request. All data are open (USGS ScienceBase, GNS Science, Copernicus, TerraClimate), and the code will be released with the paper.

Thank you for considering our proposal. We look forward to hearing from you.

Kind regards,

Quang-Vinh Dang
British University Vietnam, Hung Yen, Vietnam
vinh.dq4@buv.edu.vn | ORCID: 0000-0002-3877-8024
