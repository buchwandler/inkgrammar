# SAS-1968 annealed fitting benchmark

## Decision

The custom annealed search is **not promoted**. It met the guide's candidate-distance comparison on 3 of 4 families, the four-family robustness proxy, and the 10x runtime limit. However, none of the 100 fits passed all non-visual promotion gates. Topology, candidate-to-target p95, supervised endpoint RMS, and transform-saturation gates all failed in every run. Private visual review was not performed. Existing promotion thresholds were not changed.

The tracked SAS project now uses the proposed objective-v4 annealed configuration as an experimental default. The benchmark used a smaller, budget-matched search configuration described below. No seed grammar or primitive was promoted.

## Scope and method

The benchmark used the four topology-reviewed families `sas-underturn-01`, `sas-open-oval-01`, `sas-hump-01`, and `sas-ascender-loop-01`, with seeds 1968 through 1972. It ran five methods for each family and seed, for 100 fits total:

- v4 baseline spline fitting
- uniform random seed restarts followed by v4 refinement
- SciPy differential-evolution seed search followed by v4 refinement
- SciPy dual-annealing seed search followed by v4 refinement
- InkGrammar simulated annealing followed by v4 refinement

The global search budget was 25 coarse family-fit evaluations per run. Differential evolution used 24 evaluations under its population-generation budget. The custom annealer used 16 steps, 8 temperature-calibration samples, and the initial evaluation. Random restarts used the same 25-evaluation budget. Dual annealing used `maxfun=25`. All global methods retained the top two coarse candidates, refined them for two cycles at 64 samples, then refined the best candidate for four cycles at 128 samples. Coarse refinement used one cycle at 32 samples. The baseline used four v4 cycles. Transform fitting used 30 function evaluations and one IRLS iteration for this benchmark.

These are practical benchmark overrides, not the tracked annealing defaults. The tracked configuration uses 240 steps, 32 calibration samples, top-K 4, and 2/4/10 refinement cycles. Results below do not claim the full-budget defaults were benchmarked.

Only each family's fit members entered candidate fitting, ranking, and seed selection. No family had validation members, so validation metrics were unavailable and were not used. Hold-out values below are post-fit diagnostics only. The best run in each method/family group was selected by its fit objective, never by hold-out.

The benchmark read the already-materialized private SAS data from ignored `work/`. Only scalar metrics are recorded here. This report contains no source images, derived masks, per-evaluation histories, or private file paths.

## Results across five seeds

Training is shown as best / median / worst fit objective. Candidate and target distances, endpoint RMS, hold-out diagnostic, runtime, and evaluation counts are medians across the five runs. The topology and non-visual columns count passing runs. `Coarse/total evaluations` separates the global seed-search calls from the optimizer's aggregate count.

| Family        | Method                 | Training best / median / worst | Candidate / target p95 (px) | Endpoint RMS (px) | Coverage | Topology | Non-visual gates | Assignment stable | No saturation | Coarse / total evals | Hold-out median, diagnostic | Median seconds |
| ------------- | ---------------------- | -----------------------------: | --------------------------: | ----------------: | -------: | -------: | ---------------: | ----------------: | ------------: | -------------------: | --------------------------: | -------------: |
| underturn     | v4 baseline            |          2.705 / 2.705 / 2.705 |                 5.87 / 1.22 |              7.26 |    1.000 |      0/5 |              0/5 |               5/5 |           0/5 |               0 / 12 |                       11.27 |           1.99 |
| underturn     | random restarts        |          1.771 / 1.971 / 2.027 |                 4.03 / 1.05 |              5.43 |    1.000 |      0/5 |              0/5 |               5/5 |           0/5 |              25 / 40 |                       12.20 |           6.79 |
| underturn     | differential evolution |          1.820 / 1.958 / 2.026 |                 4.01 / 1.01 |              5.31 |    1.000 |      0/5 |              0/5 |               5/5 |           0/5 |              24 / 39 |                       12.05 |           6.62 |
| underturn     | SciPy dual annealing   |          1.937 / 2.021 / 2.096 |                 4.13 / 1.02 |              5.56 |    1.000 |      0/5 |              0/5 |               5/5 |           0/5 |              25 / 40 |                       12.14 |           6.99 |
| underturn     | InkGrammar annealing   |          1.623 / 1.995 / 2.129 |                 4.08 / 1.03 |              5.45 |    1.000 |      0/5 |              0/5 |               5/5 |           0/5 |              25 / 39 |                       12.17 |           7.24 |
| open oval     | v4 baseline            |          3.880 / 3.880 / 3.880 |                 4.47 / 5.59 |              5.55 |    1.000 |      0/5 |              0/5 |               0/5 |           0/5 |                0 / 8 |                       13.23 |           1.98 |
| open oval     | random restarts        |          3.648 / 3.695 / 3.858 |                 3.90 / 6.41 |              5.91 |    1.000 |      0/5 |              0/5 |               5/5 |           0/5 |              25 / 36 |                       13.84 |           9.99 |
| open oval     | differential evolution |          3.686 / 3.726 / 3.806 |                 3.83 / 6.55 |              5.87 |    1.000 |      0/5 |              0/5 |               5/5 |           0/5 |              24 / 35 |                       13.80 |          12.98 |
| open oval     | SciPy dual annealing   |          3.569 / 3.827 / 3.927 |                 4.17 / 6.07 |              6.35 |    1.000 |      0/5 |              0/5 |               5/5 |           0/5 |              25 / 36 |                       13.50 |          10.35 |
| open oval     | InkGrammar annealing   |          3.640 / 3.713 / 3.889 |                 3.69 / 6.91 |              6.05 |    1.000 |      0/5 |              0/5 |               5/5 |           0/5 |              25 / 35 |                       14.29 |          10.10 |
| hump          | v4 baseline            |          5.772 / 5.772 / 5.772 |                 8.90 / 8.23 |             30.52 |    1.000 |      0/5 |              0/5 |               5/5 |           0/5 |               0 / 12 |                        7.79 |           1.90 |
| hump          | random restarts        |          5.437 / 5.879 / 5.903 |                10.14 / 7.69 |             29.94 |    1.000 |      0/5 |              0/5 |               5/5 |           0/5 |              25 / 40 |                        7.26 |           8.74 |
| hump          | differential evolution |          5.959 / 6.062 / 6.957 |                10.00 / 7.41 |             29.70 |    1.000 |      0/5 |              0/5 |               4/5 |           0/5 |              24 / 39 |                        7.19 |           8.17 |
| hump          | SciPy dual annealing   |          5.794 / 5.880 / 6.107 |                10.20 / 7.46 |             29.89 |    1.000 |      0/5 |              0/5 |               5/5 |           0/5 |              25 / 40 |                        7.26 |           6.88 |
| hump          | InkGrammar annealing   |          5.737 / 5.797 / 5.921 |                10.03 / 7.81 |             30.06 |    1.000 |      0/5 |              0/5 |               5/5 |           0/5 |              25 / 40 |                        7.45 |           7.79 |
| ascender loop | v4 baseline            |       11.401 / 11.401 / 11.401 |                32.40 / 3.55 |             36.85 |    1.000 |      0/5 |              0/5 |               5/5 |           0/5 |                0 / 8 |                       14.03 |           1.57 |
| ascender loop | random restarts        |       11.257 / 11.389 / 11.495 |                32.01 / 2.93 |             37.57 |    1.000 |      0/5 |              0/5 |               5/5 |           0/5 |              25 / 36 |                       15.53 |           7.54 |
| ascender loop | differential evolution |       11.322 / 11.420 / 11.471 |                32.02 / 2.92 |             37.45 |    1.000 |      0/5 |              0/5 |               5/5 |           0/5 |              24 / 35 |                       15.68 |           6.26 |
| ascender loop | SciPy dual annealing   |       11.224 / 11.321 / 11.374 |                31.99 / 2.97 |             36.49 |    1.000 |      0/5 |              0/5 |               5/5 |           0/5 |              25 / 36 |                       14.91 |           6.18 |
| ascender loop | InkGrammar annealing   |       11.305 / 11.408 / 11.449 |                31.98 / 2.95 |             37.51 |    1.000 |      0/5 |              0/5 |               5/5 |           0/5 |              25 / 36 |                       15.59 |           6.49 |

## Guide threshold outcomes

For the candidate-distance comparison, each method's run with the lowest fit objective was selected. The comparison tolerance for target-to-candidate p95 was 2 px, matching the unchanged absolute promotion gate value as a comparison allowance.

| Family        | Baseline candidate / target p95 | Annealed candidate / target p95 | Candidate improved | Target change within 2 px | Quality comparison |
| ------------- | ------------------------------: | ------------------------------: | ------------------ | ------------------------- | ------------------ |
| underturn     |                     5.87 / 1.22 |                     3.05 / 1.05 | yes                | yes                       | pass               |
| open oval     |                     4.47 / 5.59 |                     3.32 / 6.91 | yes                | yes, +1.32 px             | pass               |
| hump          |                     8.90 / 8.23 |                     9.83 / 7.39 | no                 | yes                       | fail               |
| ascender loop |                    32.40 / 3.55 |                    32.05 / 2.86 | yes                | yes                       | pass               |

The quality-comparison threshold passed for 3/4 families. This does not imply the absolute geometry gates passed. Across all 100 runs, the automatic-gate true counts were: assigned coverage 100/100, assignment stability 94/100, target-to-candidate p95 25/100, candidate-to-target p95 0/100, supervised endpoint RMS 0/100, topology match 0/100, and no transform saturation 0/100. No method/family group had all non-visual gates pass in any of its five runs.

The custom method met the runtime threshold in all four families. Median custom-to-baseline runtime ratios were 3.27x for underturn, 5.10x for open oval, 4.20x for hump, and 4.87x for ascender loop. Timings include fit reporting and private overlay generation and are local measurements.

For a reproducible basin proxy, runs were considered in the same basin when their selected-path signature matched the modal signature and their six-control-point RMS distance from the five-run median was at most 2 px. The custom method met this proxy in 5/5 runs for each family. Maximum observed control-point RMS spread was 0.055 px, 0.080 px, 0.106 px, and 0.021 px, respectively. This proxy does not replace private visual review.

Hold-out medians are present in the results table for diagnosis only. They were not read by the search, used for seed selection, or used to choose settings. The SciPy search success/status fields and the custom annealing summaries are bounded in the ignored per-run reports; no per-evaluation traces were added to this report.

## Promotion and follow-up

No primitive is promoted. The absolute candidate-distance, supervised endpoint, topology, and transform-saturation gates did not pass. Promotion remains pending private visual review, and no existing threshold was weakened. The experiment supports retaining the custom backend for further evaluation, but not changing canonical grammar state based on these results. The hump candidate-distance result did not improve, and the ascender-loop candidate-distance remains substantially above the 2 px gate.
