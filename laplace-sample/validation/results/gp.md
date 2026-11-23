# gp: laplace-sample vs CmdStan 2.39.0

Reference: CmdStan, 4 x 25000 draws (seed 1). Runs: 4 x 1000 warmup + 1000 draws.

## nuts — max |z| = 2.59 (PASS), max R-hat 1.0031, min ESS 1915

|variable | ref_mean|    mean|  z_mean| ref_sd|     sd|   z_sd|   rhat| ess_bulk|
|:--------|--------:|-------:|-------:|------:|------:|------:|------:|--------:|
|lp__     |  16.0684| 16.0494| -0.6429| 1.2439| 1.2696| 0.6923| 1.0013| 1914.756|
|alpha    |   0.8365|  0.8348| -0.2915| 0.2577| 0.2588| 0.1694| 1.0031| 2368.229|
|rho      |   1.2431|  1.2417| -0.2270| 0.2753| 0.2764| 0.2434| 1.0020| 2358.178|
|sigma    |   0.2702|  0.2718|  2.5853| 0.0369| 0.0384| 1.9604| 0.9997| 4466.247|

## mala — max |z| = 2.44 (PASS), max R-hat 1.0036, min ESS 837

|variable | ref_mean|    mean|  z_mean| ref_sd|     sd|    z_sd|   rhat|  ess_bulk|
|:--------|--------:|-------:|-------:|------:|------:|-------:|------:|---------:|
|lp__     |  16.0684| 16.1037|  1.0439| 1.2439| 1.1730| -2.4359| 1.0036| 1099.0969|
|alpha    |   0.8365|  0.8307| -0.6492| 0.2577| 0.2533| -0.5547| 1.0024|  836.8791|
|rho      |   1.2431|  1.2291| -1.5724| 0.2753| 0.2696| -1.1466| 1.0025|  916.4379|
|sigma    |   0.2702|  0.2708|  0.5140| 0.0369| 0.0366| -0.3263| 1.0017| 1006.6136|

## Cost (sampling phase only)

`grads`: CmdStan n_leapfrog__ (one gradient each), laplace-sample n_grad__. `us_per_grad`: sampling wall time / grads, per chain summed, sampler included.

|sampler             | min_ess_bulk| grads| ess_per_grad| us_per_grad|
|:-------------------|------------:|-----:|------------:|-----------:|
|CmdStan NUTS        |         1840| 22358|      0.08230|       164.1|
|laplace-sample nuts |         1915| 14876|      0.12871|       207.0|
|laplace-sample mala |          837|  4000|      0.20922|       207.5|

