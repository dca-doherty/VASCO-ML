# Statistical Analysis

All statistical analyses were conducted in Python using the `statsmodels`, `scipy`, and `scikit-learn` libraries. Given that both study hypotheses were *a priori* and based on our prior findings, one-tailed probability values were used to determine statistical significance in all analyses described below.

## Machine Learning Model Discrimination

Machine learning model discrimination was evaluated through 5-fold stratified cross-validation using manually classified transients as the reference standard. Final metrics reflect cross-validated out-of-fold values: area under the ROC curve (AUC) with standard deviation across folds, sensitivity (true positives / (true positives + false negatives)), and specificity (true negatives / (true negatives + false positives)). Sensitivity and specificity were also derived from pooled out-of-fold predictions.

The final model achieved AUC = 0.809 +/- 0.041, sensitivity = 0.709, and specificity = 0.714. Training labels comprised 250 image pairs classified by expert visual review (134 real transients, 116 plate defects).

## Decile Stratification

Following cross-validation, the model was deployed on the full catalog of 107,875 transient candidates. For the 99,933 candidates falling on the 276 strict POSS-I observation dates (out of 370 total), transients were distributed into 10 equal-sized bins (deciles) of increasing ML-derived probability. Decile 0 (D0) contained the lowest-probability transients; Decile 9 (D9) contained the highest. Each decile contained approximately 9,994 transients.

## Shadow Deficit Analysis

The shadow deficit quantifies whether transients are underrepresented in Earth's geometric shadow at geostationary orbit (35,786 km). Each transient's angular separation from the geocentric anti-solar direction was computed using the observation date. Transients with separation less than 8.50 degrees were classified as in-shadow; all others were classified as sunlit.

The expected shadow fraction under random geometric distribution was established by a plate-aware Monte Carlo procedure. Individual plates were identified by spatial clustering of transient positions per observation date (complete linkage, 15-degree threshold), yielding 626 plates. For each plate, 100 random sky positions were generated within the plate field of view using the plate center coordinates and observation time, with no reference to actual transient positions. Of 62,600 total control positions, 433 (0.692%) fell within Earth's geometric shadow at GEO, establishing the expected shadow fraction.

For each of the 10 probability deciles, the observed shadow fraction was compared to the 0.692% expected fraction using a one-sided binomial test. The shadow deficit at each decile was expressed as the percentage reduction relative to this control rate. Because ten simultaneous binomial tests were conducted across deciles, a Bonferroni-corrected significance threshold of p < 0.005 was applied.

To test whether the highest-probability transients showed a disproportionately larger shadow deficit compared to lower-probability transients, the shadow fraction of the upper decile (D9) was compared to the pooled shadow fraction of all other deciles (D0 through D8) using a two-proportion z-test. Follow-up Bonferroni-corrected pairwise two-proportion z-tests between D9 and each lower decile were conducted as sensitivity analyses (9 tests, alpha = 0.0056).

## Nuclear Testing Association Analysis

Nuclear testing association analyses were restricted to US nuclear tests (Nevada Test Site, 435 km from Mount Palomar), comprising 95 unique test dates during the study period. Of the 370 POSS-I observation dates, 26 fell within a +/- 1 day window of any US test.

The primary hypothesis test used a non-parametric permutation approach with observation dates as the independent unit of analysis (n = 370). For each date, the sum of ML-assigned probabilities across all transients on that date was computed, yielding a probability-weighted transient count that represents the estimated number of real transients. The mean probability-weighted count for nuclear window dates was compared to the mean for non-nuclear dates. Statistical significance was assessed via 10,000 random permutations of the nuclear window labels. A confirmatory Mann-Whitney U test was also conducted, leveraging greater statistical power for detecting shifts in continuous distributions.

To parallel the shadow deficit analysis, the percentage of D9 transients falling within a nuclear window was compared to the pooled percentage across D0 through D8 using a two-proportion z-test.

To assess temporal specificity, the primary permutation test was repeated at each lag from 3 days before a nuclear test date to 3 days after. For each significant lag, effect size was expressed as the unadjusted ratio of the mean probability-weighted transient count on nuclear-window dates to the mean on non-nuclear dates.

## Compact Substructures

Within the highest-probability decile (D9), spatial groupings of transients within 10 arcmin on the same observation night were identified via complete-linkage hierarchical clustering. Counts were reported for doublets (2 transients), triplets (3 transients), and larger groupings (4 or more transients).
