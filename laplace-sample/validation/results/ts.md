# ts: laplace-sample vs CmdStan 2.39.0

Reference: CmdStan, 4 x 25000 draws (seed 1). Runs: 4 x 1000 warmup + the draws shown.
Verdict: did-not-converge unless every quantity has R-hat < 1.01 and bulk ESS > 400; then pass if every |z| < 4.

## nuts — pass (4 x 1000 draws): max |z| = 1.49, max R-hat 1.0023, min bulk ESS 2048

|variable |  ref_mean|      mean|  z_mean| ref_sd|     sd|    z_sd|   rhat| ess_bulk|
|:--------|---------:|---------:|-------:|------:|------:|-------:|------:|--------:|
|lp__     | -203.8914| -203.9088| -0.6008| 1.2171| 1.2581|  1.4100| 1.0015| 2047.574|
|alpha    |    0.2164|    0.2168|  0.4887| 0.0351| 0.0355|  0.7504| 1.0005| 2287.466|
|phi[1]   |    0.4880|    0.4868| -1.1061| 0.0508| 0.0507| -0.1701| 1.0003| 2358.947|
|sigma    |    0.4793|    0.4797|  1.2275| 0.0197| 0.0202|  1.4872| 1.0023| 4779.986|

## mala — pass (4 x 1000 draws): max |z| = 1.92, max R-hat 1.0055, min bulk ESS 496

|variable |  ref_mean|      mean|  z_mean| ref_sd|     sd|   z_sd|   rhat|  ess_bulk|
|:--------|---------:|---------:|-------:|------:|------:|------:|------:|---------:|
|lp__     | -203.8914| -203.9008| -0.2335| 1.2171| 1.2828| 1.6089| 1.0022| 1015.0495|
|alpha    |    0.2164|    0.2181|  1.0467| 0.0351| 0.0367| 1.9190| 1.0055|  517.1176|
|phi[1]   |    0.4880|    0.4842| -1.6462| 0.0508| 0.0514| 0.4145| 1.0047|  496.0556|
|sigma    |    0.4793|    0.4798|  0.7554| 0.0197| 0.0197| 0.0484| 1.0032|  870.2485|

## Cost (sampling phase only)

`grads`: CmdStan n_leapfrog__ (one gradient each), laplace-sample n_grad__. `us_per_grad`: sampling wall time / grads, per chain summed, sampler included.

|sampler             | min_ess_bulk| grads| ess_per_grad| us_per_grad|
|:-------------------|------------:|-----:|------------:|-----------:|
|CmdStan NUTS        |         2035| 21474|      0.09475|        23.7|
|laplace-sample nuts |         2048| 14926|      0.13718|        26.8|
|laplace-sample mala |          496|  4000|      0.12401|        67.5|

