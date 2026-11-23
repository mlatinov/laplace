# ts: laplace-sample vs CmdStan 2.39.0

Reference: CmdStan, 4 x 25000 draws (seed 1). Runs: 4 x 1000 warmup + 1000 draws.

## nuts — max |z| = 1.49 (PASS), max R-hat 1.0023, min ESS 2048

|variable |  ref_mean|      mean|  z_mean| ref_sd|     sd|    z_sd|   rhat| ess_bulk|
|:--------|---------:|---------:|-------:|------:|------:|-------:|------:|--------:|
|lp__     | -203.8914| -203.9088| -0.6008| 1.2171| 1.2581|  1.4100| 1.0015| 2047.574|
|alpha    |    0.2164|    0.2168|  0.4887| 0.0351| 0.0355|  0.7504| 1.0005| 2287.466|
|phi[1]   |    0.4880|    0.4868| -1.1061| 0.0508| 0.0507| -0.1701| 1.0003| 2358.947|
|sigma    |    0.4793|    0.4797|  1.2275| 0.0197| 0.0202|  1.4872| 1.0023| 4779.986|

## mala — max |z| = 1.83 (PASS), max R-hat 1.0067, min ESS 545

|variable |  ref_mean|      mean|  z_mean| ref_sd|     sd|    z_sd|   rhat| ess_bulk|
|:--------|---------:|---------:|-------:|------:|------:|-------:|------:|--------:|
|lp__     | -203.8914| -203.9236| -0.7921| 1.2171| 1.2837|  1.6673| 1.0030| 962.3373|
|alpha    |    0.2164|    0.2189|  1.5856| 0.0351| 0.0366|  1.8329| 1.0067| 569.1051|
|phi[1]   |    0.4880|    0.4849| -1.4215| 0.0508| 0.0520|  0.9593| 1.0059| 545.1302|
|sigma    |    0.4793|    0.4795|  0.3007| 0.0197| 0.0196| -0.1675| 1.0018| 899.7938|

## Cost (sampling phase only)

`grads`: CmdStan n_leapfrog__ (one gradient each), laplace-sample n_grad__. `us_per_grad`: sampling wall time / grads, per chain summed, sampler included.

|sampler             | min_ess_bulk| grads| ess_per_grad| us_per_grad|
|:-------------------|------------:|-----:|------------:|-----------:|
|CmdStan NUTS        |         2035| 21474|      0.09475|        23.8|
|laplace-sample nuts |         2048| 14926|      0.13718|        24.1|
|laplace-sample mala |          545|  4000|      0.13628|        60.0|

