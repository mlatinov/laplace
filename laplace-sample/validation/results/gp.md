# gp: laplace-sample vs CmdStan 2.39.0

Reference: CmdStan, 4 x 25000 draws (seed 1). Runs: 4 x 1000 warmup + the draws shown.
Verdict: did-not-converge unless every quantity has R-hat < 1.01 and bulk ESS > 400; then pass if every |z| < 4.

## nuts — pass (4 x 1000 draws): max |z| = 2.59, max R-hat 1.0031, min bulk ESS 1915

|variable | ref_mean|    mean|  z_mean| ref_sd|     sd|   z_sd|   rhat| ess_bulk|
|:--------|--------:|-------:|-------:|------:|------:|------:|------:|--------:|
|lp__     |  16.0684| 16.0494| -0.6429| 1.2439| 1.2696| 0.6923| 1.0013| 1914.756|
|alpha    |   0.8365|  0.8348| -0.2915| 0.2577| 0.2588| 0.1694| 1.0031| 2368.229|
|rho      |   1.2431|  1.2417| -0.2270| 0.2753| 0.2764| 0.2434| 1.0020| 2358.178|
|sigma    |   0.2702|  0.2718|  2.5853| 0.0369| 0.0384| 1.9604| 0.9997| 4466.247|

## mala — pass (4 x 1000 draws): max |z| = 1.60, max R-hat 1.0059, min bulk ESS 836

|variable | ref_mean|    mean|  z_mean| ref_sd|     sd|    z_sd|   rhat|  ess_bulk|
|:--------|--------:|-------:|-------:|------:|------:|-------:|------:|---------:|
|lp__     |  16.0684| 16.0907|  0.6039| 1.2439| 1.1989| -1.5891| 1.0059| 1021.7526|
|alpha    |   0.8365|  0.8238| -1.4589| 0.2577| 0.2491| -1.0989| 1.0047|  835.9365|
|rho      |   1.2431|  1.2288| -1.5602| 0.2753| 0.2662| -1.6014| 1.0041|  843.1715|
|sigma    |   0.2702|  0.2712|  0.8650| 0.0369| 0.0373|  0.5488| 1.0020| 1028.9262|

## Cost (sampling phase only)

`grads`: CmdStan n_leapfrog__ (one gradient each), laplace-sample n_grad__. `us_per_grad`: sampling wall time / grads, per chain summed, sampler included.

|sampler             | min_ess_bulk| grads| ess_per_grad| us_per_grad|
|:-------------------|------------:|-----:|------------:|-----------:|
|CmdStan NUTS        |         1840| 22358|      0.08230|       162.8|
|laplace-sample nuts |         1915| 14876|      0.12871|       148.6|
|laplace-sample mala |          836|  4000|      0.20898|       152.5|

