# Level 1: pick a sampler

> **Experimental in laplace 0.3.0.** Draft wiki page. Stan's NUTS through
> CmdStan stays the default way to run a model.

Run a laplace model with a sampler other than CmdStan's, without writing
anything new:

```sh
laplace sample model.laplace --data data.json --sampler mala --seed 1
```

```r
fit <- cmdlaplacer::laplace_sample("model.laplace", data = data, sampler = "mala")
fit$summary()   # an ordinary cmdstanr fit
```

`laplace sample` compiles the model (with `laplace build` for a `.laplace`
file, then BridgeStan), runs four chains in parallel, and writes one
CmdStan-style CSV per chain. `cmdstanr::as_cmdstan_fit()` reads them.

## The built-in samplers

| `--sampler` | What it is | Adapts in warmup | `--set` settings |
| --- | --- | --- | --- |
| `nuts` (default) | No-U-Turn sampler (nuts-rs, with nuts-rs's own warmup adaptation) | step size, diagonal metric | `step_size`, `target_accept`, `max_depth` |
| `mala` | Metropolis-adjusted Langevin | step size (to 0.574), diagonal metric | `step_size`, `target_accept`, `adapt` |
| `hmc` | Static HMC, path length jittered | step size (to 0.8), diagonal metric | the same plus `steps`, `jitter` |
| `rwm` | Random-walk Metropolis | step size (to 0.234), diagonal metric | `step_size`, `target_accept`, `adapt` |
| `path/to/k.lapkernel` | your own kernel (Level 3) | whatever it tags `@adapt` | its tuning values, `adapt`, `target_accept` |
| `langevin::mala` | a kernel from an installed kernel package | as the kernel says | as the kernel says |
| `file.lapsampler` | a composed sampler (Level 2) | per kernel, as the file says | none (they live in the file) |

```sh
laplace sample model.laplace --sampler hmc --set steps=20 --set target_accept=0.9
laplace sample model.laplace --sampler mala --set step_size=0.05 --set adapt=false
```

Adaptation is frozen when warmup ends. Saved draws are never taken with a
tuning that is still moving.

## Run settings

`--chains` (4), `--parallel-chains`, `--warmup` (1000), `--draws` (1000),
`--thin` (1), `--seed`, `--init` (a radius, or JSON files), `--output-dir`.
`laplace-sample sample --help` lists them all.

## What the CSV tells you

- **Diagnostic columns:** the usual columns. One that does not apply to
  the sampler is `NaN`, so `treedepth__` is `NaN` for MALA. cmdlaplacer
  says which apply.
- **Cost columns:** `n_grad__` (gradient evaluations) and `n_logp__`
  (density-only evaluations, e.g. by `rwm`). Use them to compare
  efficiency, together with the run time printed per chain.
- **Provenance comments:** what produced the run.

  ```text
  # laplace_sampler = mala (sha256: ...)
  # laplace_kernel_rest = mala 0.3.0 sha256:...
  # laplace_settings = adapt:true step_size:1 target_accept:0.574
  ```

## Which sampler?

NUTS, unless you have measured a reason. A gradient-based kernel without
NUTS's adaptive path length (MALA, fixed-length HMC) needs many more
iterations on correlated posteriors, and RWM even more. Before trusting a
sampler on your model, check it on the reference suite:

```sh
laplace sampler test mala
```

See [Level 2](Level-2-Compose-kernels.md) for using different samplers on
different parameters.
