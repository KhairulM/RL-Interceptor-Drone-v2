# F14 Stage A: 13-stage rule-selection baseline

All evaluations use common thesis episode-ending semantics. Pursuer OOB is not terminal or an evader win; it is reported separately.
G1 = equal-weight mean of strong-set J and campaign-balanced cross-seed/campaign J. G2 = own-rules exploiter J minus F8 own-rules exploiter J.
Both gaps are shown before their sum; positive is better than F8. Three seeds per family; descriptive seed means/ranges.

## Per-seed scores at 0.10 m

| Family | Campaign | Strong J | Cross J | G1 | Own-rules J | Own exploitability | Thesis-lure J | Pursuer OOB (own) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| f8 | thesis-F8b-roncero-s0 | 0.858 | 0.882 | 0.870 | 0.662 | 0.338 | 0.369 | 0.283 |
| f8 | thesis-F8b-roncero-s100 | 0.876 | 0.936 | 0.906 | 0.294 | 0.706 | 0.487 | 0.092 |
| f8 | thesis-F8b-roncero-s200 | 0.786 | 0.956 | 0.871 | 0.925 | 0.075 | 0.424 | 0.914 |
| f10 | thesis-F10-nofix-s0 | 0.831 | 0.888 | 0.860 | 0.828 | 0.172 | 0.828 | 0.087 |
| f10 | thesis-F10-nofix-s100 | 0.608 | 0.806 | 0.707 | 0.192 | 0.808 | 0.192 | 0.144 |
| f10 | thesis-F10-nofix-s200 | 0.701 | 0.870 | 0.785 | 0.403 | 0.597 | 0.403 | 0.645 |
| f11 | thesis-F11v2d-s0 | 0.873 | 0.934 | 0.903 | 0.725 | 0.275 | 0.688 | 0.445 |
| f11 | thesis-F11v2d-s100 | 0.679 | 0.815 | 0.747 | 0.984 | 0.016 | 0.307 | 0.023 |
| f11 | thesis-F14-v2d-short-s200 | 0.904 | 0.884 | 0.894 | 0.918 | 0.082 | 0.800 | 0.066 |

## Means and gaps vs F8 (0.10 m)

| Candidate | G1 strong component gap | G1 cross-seed component gap | G1 total gap | G2 own-rules exploitability gap | G1+G2 |
|---|---:|---:|---:|---:|---:|
| f10 | -0.126 | -0.070 | -0.098 | -0.152 | -0.250 |
| f11 | -0.021 | -0.048 | -0.034 | +0.249 | +0.215 |

**Selected rule set:** v2d (larger equal-weight sum; tie defaults to v2d).
