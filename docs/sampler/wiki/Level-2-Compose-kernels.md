# Level 2: compose kernels

> **Experimental in laplace 0.3.0.** Draft wiki page.

A `.lapsampler` file says which kernel updates which parameters:

```text
sampler hybrid {
  library { import langevin }                       // optional: kernel packages
  kernel {
    block(beta)              : nuts()
    block(sigma, tau) as sc  : adapt(langevin::mala, target_accept = 0.6)
    rest                     : adapt(rwm())
  }
  warmup { iterations = 1000; }                     // optional; --warmup overrides
  draws  { iterations = 1000; thin = 1; }           // optional; --draws/--thin override
}
```

```sh
laplace sampler check hybrid.lapsampler --model model.laplace --data data.json
laplace sample model.laplace --data data.json --sampler hybrid.lapsampler
```

Each iteration updates the blocks in order. Each block's kernel moves only
its own parameters, with the others held at their current values (a Gibbs
sweep).

## Blocks

- **Whole parameters only:** `block(beta)`, not `block(beta[1:3])`.
- **One block per parameter:** a parameter may appear in one block only.
- **Coverage is checked:** every parameter must be in some block. A
  missing one is an error, so end with `rest : <kernel>` to take whatever
  is left.
- **Labels:** a block's label is `as <label>`, or its parameter names
  joined with `_`. Labels name the CSV's per-block columns
  (`accept_stat_sc__`, `stepsize_sc__`, ...).
- **Constrained parameters are fine.** A simplex block is updated on its
  unconstrained coordinates, so `simplex[4]` has 3.

## Kernels and combinators

| Expression | Meaning |
| --- | --- |
| `nuts(...)`, `mala(...)`, `hmc(...)`, `rwm(...)` | a built-in, with settings: nuts `step_size, target_accept, max_depth`; mala/rwm `step_size`; hmc `step_size, steps, jitter` |
| `./kernels/k.lapkernel(eps = 0.1)` | a local kernel (Level 3), with tuning values; the path is relative to the `.lapsampler` |
| `langevin::mala(eps = 0.1)` | a kernel from an imported kernel package |
| `adapt(k, target_accept = 0.6)` | adapt `k`'s tagged step size and metric during warmup, then freeze them |
| `seq(k1, k2, ...)` | run each in turn |
| `repeat(n, k)` | run `k` n times |
| `mix(1: k1, 3: k2)` | pick one at random by weight each iteration |

**Adaptation is explicit at Level 2.** `rwm()` runs with its given step
size, and `adapt(rwm())` adapts it. `nuts()` adapts itself and cannot sit
under `adapt()` or `mix()`.

**Imports:**

- `import langevin` brings an installed kernel package into scope (`laplace
  add langevin --path ...`, or `--git ... --tag ...`).
- Versions come only from `laplace.toml` and `laplace.lock`, never from
  the `.lapsampler`.
- The lock is the one in the current directory, as for every `laplace`
  command.

## When composition helps, and when it doesn't

Composition is a tool for geometry NUTS handles poorly. A block may need a
different kind of move, have a much cheaper gradient, or (with `mix`) need
occasional large jumps. It is not a way to make a well-behaved model
faster. In every hybrid we measured on such models, plain NUTS was more
efficient:

| Model | ESS/grad vs NUTS | ESS/s vs NUTS |
| --- | --- | --- |
| Eight schools, NUTS + MALA | 0.31 | 0.29 |
| Correlated Gaussian split in two blocks, NUTS + MALA | 0.60 | 0.54 |
| Simplex model, MALA + NUTS | 0.24 | 0.14 |
| Survival model, MALA + Barker | 0.24 | 0.22 |

Block updates hold the other blocks fixed. When parameters in different
blocks are strongly correlated, each sweep moves only a little, and the
chain needs many more iterations. **Put correlated parameters in the same
block.** Measure with `laplace sampler test hybrid.lapsampler` (the ratio
to plain NUTS is in the table) before switching a model to a hybrid.

## The CSV

With several blocks:

- every block gets its own columns (`accept_stat_<label>__`, and so on);
- the standard columns are `NaN` where one value cannot describe several
  kernels;
- `divergent__` is any block's divergence, and `n_grad__`/`n_logp__` are
  totals;
- one `# laplace_kernel_<label> = <expression> <version> sha256:<hash>`
  line per block records exactly what ran.

One block covering the whole model writes the standard, unsuffixed
columns.
