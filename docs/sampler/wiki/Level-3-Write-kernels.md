# Level 3: write kernels

> **Experimental in laplace 0.3.0.** Draft wiki page.

A `.lapkernel` file is one MCMC transition in a small Stan-like language.
laplace-sample interprets it, and BridgeStan supplies the model's log
density and gradient. Every kernel you run is readable source, and its hash
is recorded in the CSV of every run that used it.

```sh
laplace new kernel pmala                 # a new kernel package pmala/ with kernel pmala
laplace new kernel mala --package langevin   # a package langevin/ whose first kernel is mala
laplace new kernel gibbs --kind direct   # inside a package: add a kernel ending in move()
laplace sampler check kernels/pmala.lapkernel
laplace sample model.laplace --sampler kernels/pmala.lapkernel
laplace sampler test kernels/pmala.lapkernel
```

## A complete kernel

```stan
//@laplace
//@brief Metropolis-adjusted Langevin algorithm with a diagonal metric
//@param eps Step size (adapted toward 0.574 acceptance)
//@param m Diagonal inverse metric (adapted in warmup)
kernel mala {
  tuning {
    real<lower=0> eps = 1 @adapt(step_size, target_accept = 0.574);
    vector<lower=0>[dim] m = rep_vector(1, dim) @adapt(metric);
  }
  step {
    vector[dim] mu_fwd = q + 0.5 * eps^2 * m .* grad;
    vector[dim] q_new = mu_fwd + eps * sqrt(m) .* std_normal_rng(dim);
    (real lp_new, vector[dim] g_new) = log_density_grad(q_new);
    vector[dim] mu_bwd = q_new + 0.5 * eps^2 * m .* g_new;
    vector[dim] two_var = 2 * eps^2 * m;
    propose(q_new, sum(square(q_new - mu_fwd) ./ two_var) - sum(square(q - mu_bwd) ./ two_var));
  }
}
```

## The file

| Section | Holds |
| --- | --- |
| `tuning { }` | Values the user may set (`--set eps=0.1`, `k(eps = 0.1)` in a `.lapsampler`), with defaults. A tag `@adapt(step_size, target_accept = x)` on a positive real, or `@adapt(metric)` on a positive `vector[dim]`, lets warmup adapt it. `target_accept` is required on a step-size tag. |
| `state { }` | Optional variables kept across iterations. |
| `init { }` | Optional statements run once, before the first step. |
| `functions { }` | Optional helper functions, in Stan syntax. |
| `step { }` | One transition. It must end in exactly one of `propose(q_new, c)` or `move(q_new)`. |

- **Types:** `int`, `real`, `vector[n]`, `matrix[r, c]`, `tuple(...)`, with
  optional `<lower=..., upper=...>` bounds.
- **Statements:** declarations, assignments (`=`, `+=`, ...), `if`/`else`,
  `for (i in 1:n)`, `while`, `break`, `continue`, `return`, and tuple
  destructuring `(real a, vector[dim] b) = f(...);`. As in Stan, `break`
  and `continue` act on the innermost enclosing loop only, and are an error
  outside a loop (a function body does not count as inside its caller's
  loop). `while (1) { ... if (done) break; }` is the idiom for "repeat
  until".
- **Operators:** `+ - * / % ^`, the elementwise `.*` and `./`,
  comparisons, `&&`, `||` and `!`. Indexing is 1-based, as in Stan:
  `v[i]` and `m[i, j]`, one element at a time. There are no arrays and no
  slices. Keep K vectors as the columns of a `matrix[dim, K]` and move
  them with `col(m, j)` and `m = set_col(m, j, v)` (also `row`, `set_row`,
  `rows`, `cols`, `rep_matrix`).

### Ending a step

- **`propose(q_new, c)`** is a Metropolis-Hastings proposal.
  laplace-sample draws one uniform and accepts with probability
  `min(1, exp(lp(q_new) - lp(q) + c))`, where `c` is the log proposal
  correction `log g(q | q_new) - log g(q_new | q)` (0 for a symmetric
  proposal). Getting `c` wrong is the classic bug: `laplace sampler test
  --sbc` exists to catch it, and fails a MALA with `c` dropped on every
  quantity.
- **`move(q_new)`** is a direct move, always taken. The step itself must
  leave the target invariant: an exact conditional draw, or a kernel that
  does its own accept/reject.

### What a kernel can use

| Name | What |
| --- | --- |
| `q`, `lp`, `grad`, `dim` | current position (unconstrained), its log density, its gradient, the number of coordinates the kernel updates |
| `metric` | the kernel's `@adapt(metric)` value, else all ones |
| `log_density(x)` | log density at `x` |
| `log_density_grad(x)` | `(log density, gradient)` at `x` |
| `leapfrog(q, p, eps, steps, m)` | `steps` leapfrog steps with diagonal inverse metric `m`: `(q, p, lp, grad)` |
| `std_normal_rng()`, `std_normal_rng(n)`, `normal_rng(mu, sigma)`, `uniform_rng(a, b)` | random numbers from the chain's stream |
| `normal_lpdf(y \| mu, sigma)`, `std_normal_lpdf(y)` | log densities, with constants |
| `exp`, `log`, `sqrt`, `log1p_exp`, `inv_logit`, `log_sum_exp`, `fmax`, `norm2`, `softmax`, ... | math, linear algebra, about 25 random-number generators and 25 densities: about 180 functions in all, Stan's names and meanings, elementwise on vectors where that makes sense, numerically safe forms only. Full list: [`docs/sampler/kernel-functions.md`](../kernel-functions.md) |

`laplace sampler doc <kernel>` and the registry in
`laplace-sample/src/lapkernel/builtins.rs` are the full reference. Each
built-in documents how many random numbers it draws, and in what order.

Evaluations are cached by position, so evaluating the point you then
propose costs nothing extra. A kernel that never reads `grad` never pays
for a gradient: its evaluations are density-only and are counted in the
CSV's `n_logp__`, not `n_grad__`.

## A loop that runs until it is done: slice sampling

A slice sampler (Neal 2003) needs loops whose length is not known in
advance: step a bracket out until both ends leave the slice, then draw
inside it, shrinking it on every miss, until a draw lands in the slice.
This kernel updates one coordinate at a time and shows both forms of
early exit: `while (1)` with `break` to stop stepping out, and
`continue` to try again after a rejected draw.

```stan
//@laplace
//@brief Coordinate-wise slice sampler: stepping out, then shrinkage
//@param w Initial bracket width, in the units of the unconstrained parameters
kernel slice {
  tuning {
    real<lower=0> w = 1;
  }
  step {
    vector[dim] x = q;
    real lp_x = lp;
    for (i in 1:dim) {
      real x0 = x[i];
      // The slice is every point whose log density is above log_y.
      real log_y = lp_x - exponential_rng(1);
      real lo = x0 - w * uniform_rng(0, 1);
      real hi = lo + w;

      // Step out: widen each end until it leaves the slice.
      while (1) {
        x[i] = lo;
        if (log_density(x) <= log_y) break;
        lo -= w;
      }
      while (1) {
        x[i] = hi;
        if (log_density(x) <= log_y) break;
        hi += w;
      }

      // Shrink: draw in (lo, hi). A miss pulls the end on its side of x0 in
      // and tries again; a hit is the new value of coordinate i.
      while (1) {
        x[i] = uniform_rng(lo, hi);
        lp_x = log_density(x);
        if (lp_x <= log_y) {
          if (x[i] < x0) lo = x[i]; else hi = x[i];
          continue;
        }
        break;
      }
    }
    move(x);
  }
}
```

Every shrinkage loop ends, because `x0` itself is in the slice and the
bracket closes in on it. The step ends in `move(x)`: the slice update
leaves the target invariant by itself, so there is nothing to accept or
reject. This is the kind of kernel the [Rust advice](#when-to-write-rust-instead)
below is about: it evaluates the density several times per coordinate.

**The terminal stays outside the loop.** `propose()` and `move()` may not
appear inside a `for` or `while`, and `break` does not change that, even
`while (1) { move(x); break; }`, which would run exactly once. The checker
judges where a terminal is written, not how many times a loop happens to
run. Leave the loop with `break`, then end the step after it, as `slice`
does.

## Doc comments

Lines starting `//@` before `kernel` are the kernel's documentation, shown
by `laplace doc pkg::kernel` and `laplace sampler list`:

- `//@laplace` marks the block as laplace documentation;
- `//@brief <one line>` gives the summary;
- `//@param <tuning value> <text>` describes one tuning value.

`laplace doc` also shows each tuning value as declared, with its `@adapt`
tag, and whether the step ends in `propose()` or `move()`.

## Testing and sharing

- **`laplace sampler check k.lapkernel`:** parses and type-checks the
  kernel. Editors get the same diagnostics through the
  `laplace_sample::editor` API.
- **`laplace sampler test k.lapkernel`:** runs the reference suite. It
  reads draw counts from `tests/k.test.toml`; give a kernel that "did not
  converge" more draws there.
- **`laplace sampler test k.lapkernel --sbc 200`:** simulation-based
  calibration.
- **Kernel packages:** `kind = "kernels"`, with kernels under `kernels/`,
  added to projects with `laplace add`. `laplace release` runs every
  kernel through the must-pass tier first (`--skip-sampler-tests` to opt
  out).

## When to write Rust instead

The interpreter's cost is a few µs per step, more with loops over
coordinates. On a model whose gradient takes ~100 µs that is invisible
(3.5% on the GP library model). On a density that takes ~1 µs it dominates:
the interpreted MALA makes bit-identical draws to the built-in Rust MALA at
about 1/6 of its ESS per second.

If your kernel will run on cheap models, or loops over coordinates, write
it in Rust instead:

- implement the `Kernel` trait in `laplace-sample/src/kernel.rs`, as the
  built-ins in `laplace-sample/src/kernels/` do;
- keep the `.lapkernel` as its readable specification;
- check that the two produce the same draws (`tests/equivalence.rs`).
