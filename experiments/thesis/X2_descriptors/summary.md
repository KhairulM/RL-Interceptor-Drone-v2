# X2_descriptors summary

## Diversity (Vendi score = effective number of distinct evaders)

| group | n | Vendi |
|---|---|---|
| all | 32 | 4.54 |
| rl | 22 | 3.79 |
| scripted | 10 | 1.84 |
| v10 | 6 | 2.30 |
| v11 | 6 | 2.59 |
| v9 | 10 | 3.32 |

## Redundancy (descriptor distances; RBF sigma = median distance)

- sigma: 1.874
- median_all: 1.874
- mean_consecutive_stage: 1.146
- mean_within_campaign: 1.788
- mean_across_campaign: 1.828
- nearest_neighbour_mean: 0.560

## Clusters: k-medoids, best k = 7 by silhouette

| k | silhouette |
|---|---|
| 2 | 0.265 |
| 3 | 0.318 |
| 4 | 0.373 |
| 5 | 0.414 |
| 6 | 0.455 |
| 7 | 0.461 |
| 8 | 0.397 |

| cluster | medoid | members |
|---|---|---|
| 0 | v9/evader:s1 | v9/evader:s1, v10/evader:s1, v11/evader:s1 |
| 1 | heur/evader:zigzag | heur/evader:hover, heur/evader:flee, heur/evader:weave, heur/evader:zigzag, heur/evader:fast_flee, heur/evader:hover>flee@0.5, heur/evader:flee>weave@0.5 |
| 2 | heur/evader:flee>circular@0.5 | heur/evader:circular, heur/evader:flee>circular@0.5, heur/evader:circular>hover@0.5 |
| 3 | v9/evader:s17 | v9/evader:s15, v9/evader:s17, v9/evader:s19 |
| 4 | v9/evader:s9 | v9/evader:s9, v9/evader:s11, v9/evader:s13, v11/evader:s7, v11/evader:s9, v11/evader:s11 |
| 5 | v10/evader:s9 | v10/evader:s7, v10/evader:s9, v10/evader:s11 |
| 6 | v9/evader:s5 | v9/evader:s3, v9/evader:s5, v9/evader:s7, v10/evader:s3, v10/evader:s5, v11/evader:s3, v11/evader:s5 |

## Split (held-out campaign v10)

- train (20): v9/evader:s1, v9/evader:s3, v9/evader:s5, v9/evader:s7, v9/evader:s9, v9/evader:s11, v9/evader:s13, v9/evader:s15, v9/evader:s17, v9/evader:s19, v11/evader:s1, v11/evader:s3, v11/evader:s5, v11/evader:s7, v11/evader:s9, v11/evader:s11, heur/evader:hover, heur/evader:circular, heur/evader:flee, heur/evader:weave
- held-out RL (6): v10/evader:s1, v10/evader:s3, v10/evader:s5, v10/evader:s7, v10/evader:s9, v10/evader:s11
- held-out scripted (6): heur/evader:zigzag, heur/evader:fast_flee, heur/evader:flee>circular@0.5, heur/evader:circular>hover@0.5, heur/evader:hover>flee@0.5, heur/evader:flee>weave@0.5

Figure: fig_mds.png
