# survival: laplace-sample vs CmdStan 2.39.0

Reference: CmdStan, 4 x 25000 draws (seed 1). Runs: 4 x 1000 warmup + the draws shown.
Verdict: did-not-converge unless every quantity has R-hat < 1.01 and bulk ESS > 400; then pass if every |z| < 4.

## nuts — pass (4 x 4000 draws): max |z| = 1.14, max R-hat 1.0032, min bulk ESS 1855

|variable |   ref_mean|       mean|  z_mean| ref_sd|     sd|    z_sd|   rhat| ess_bulk|
|:--------|----------:|----------:|-------:|------:|------:|-------:|------:|--------:|
|lp__     | -1044.8501| -1044.8324|  0.6509| 1.6492| 1.6297| -0.7151| 1.0002| 4506.788|
|a        |     0.0476|     0.0476|  0.0887| 0.0268| 0.0268| -0.0806| 1.0017| 1990.127|
|gamma    |     0.2501|     0.2497| -0.3455| 0.0458| 0.0459|  0.1748| 1.0032| 1967.507|
|eta0     |    -2.3230|    -2.3228|  0.0364| 0.3023| 0.3045|  0.2763| 1.0029| 1855.005|
|beta[1]  |     0.5840|     0.5846|  0.2785| 0.0917| 0.0914| -0.1895| 1.0021| 2640.729|
|beta[2]  |    -0.3625|    -0.3605|  1.1383| 0.1279| 0.1272| -0.5633| 1.0003| 6147.980|

## mala — did-not-converge (4 x 1000 draws): max |z| = 1.00, max R-hat 1.7722, min bulk ESS 6

|variable |   ref_mean|       mean|  z_mean| ref_sd|     sd|    z_sd|   rhat| ess_bulk|
|:--------|----------:|----------:|-------:|------:|------:|-------:|------:|--------:|
|lp__     | -1044.8501| -1045.1326| -0.9144| 1.6492| 1.7297|  0.6086| 1.1111|  25.6093|
|a        |     0.0476|     0.0513|  0.2861| 0.0268| 0.0315|  0.9457| 1.7722|   6.0012|
|gamma    |     0.2501|     0.2618|  0.5440| 0.0458| 0.0552|  0.8524| 1.5210|   7.1906|
|eta0     |    -2.3230|    -2.3991| -0.4713| 0.3023| 0.3914|  0.9984| 1.6775|   6.3157|
|beta[1]  |     0.5840|     0.5883|  0.1086| 0.0917| 0.1006|  0.4226| 1.5618|   7.0822|
|beta[2]  |    -0.3625|    -0.3530|  0.6723| 0.1279| 0.1225| -0.6720| 1.0561|  74.5086|

## Cost (sampling phase only)

`grads`: CmdStan n_leapfrog__ (one gradient each), laplace-sample n_grad__. `us_per_grad`: sampling wall time / grads, per chain summed, sampler included.

|sampler             | min_ess_bulk|  grads| ess_per_grad| us_per_grad|
|:-------------------|------------:|------:|------------:|-----------:|
|CmdStan NUTS        |          887|  89398|      0.00993|        95.5|
|laplace-sample nuts |         1855| 179160|      0.01035|       121.6|
|laplace-sample mala |            6|   4000|      0.00150|       120.0|

