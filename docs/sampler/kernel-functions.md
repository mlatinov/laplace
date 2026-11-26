# Kernel function library

Every function a `.lapkernel` can call without defining it. A kernel step
runs in laplace-sample's interpreter, not as compiled Stan, so a kernel
gets **this list, not Stan's whole library**. The source of truth is the
registry in `laplace-sampler-syntax/src/lapkernel/builtins.rs` (names, type
rules, docs) and `laplace-sample/src/lapkernel/natives.rs` (what each one
computes). A test checks that the two lists agree. Editors show the same
text on hover.

Names and meanings follow Stan, so a Stan user can guess them. Where Stan
has a numerically safe form of a common expression, only the safe form is
offered: write `log1p_exp(x)`, not `log(1 + exp(x))`.

**Elementwise** means that `f(vector)` applies `f` to each element and
returns a vector of the same size. In two-argument elementwise functions a
`real` broadcasts against a `vector`. `int` arguments are promoted to
`real` everywhere.

**Out of domain** gives NaN (IEEE behaviour), not an error. A NaN that
reaches `propose()` rejects the proposal. A NaN position given to `move()`
stops the run.

## Context, randomness, densities, ending a step

These are unchanged. See `docs/sampler/wiki/Level-3-Write-kernels.md`.

| Name | What |
| --- | --- |
| `q`, `lp`, `grad`, `dim`, `metric` | chain state and the adapted metric |
| `log_density(x)`, `log_density_grad(x)`, `leapfrog(q, p, eps, steps, m)` | the target |
| `std_normal_rng()`, `std_normal_rng(n)`, `normal_rng(mu, sigma)`, `uniform_rng(a, b)` | random numbers |
| `normal_lpdf(y \| mu, sigma)`, `std_normal_lpdf(y)` | log densities |
| `propose(q_new, log_correction)`, `move(q_new)` | end the step |

## Elementwise, one argument: `real f(real)`, `vector f(vector)`

| Function | Value | How it is computed / edge cases |
| --- | --- | --- |
| `exp(x)` | e^x | |
| `log(x)` | ln x | -inf at 0, NaN below |
| `sqrt(x)` | √x | NaN below 0 |
| `square(x)` | x² | |
| `abs(x)` | \|x\| | |
| `inv(x)` | 1/x | ±inf at 0 |
| `inv_sqrt(x)` | 1/√x | |
| `inv_square(x)` | 1/x² | |
| `expm1(x)` | eˣ − 1 | accurate near 0 |
| `log1p(x)` | ln(1 + x) | accurate near 0; NaN below −1 |
| `log1m(x)` | ln(1 − x) | `log1p(-x)`; NaN above 1 |
| `log1p_exp(x)` | ln(1 + eˣ) (softplus) | `x + log1p(exp(-x))` for x > 0, else `log1p(exp(x))`: never overflows |
| `log1m_exp(x)` | ln(1 − eˣ), x ≤ 0 | `log(-expm1(x))` for x > −ln 2, else `log1p(-exp(x))` (Mächler 2012); −inf at 0, NaN above |
| `inv_logit(x)` | 1 / (1 + e⁻ˣ) | only ever exponentiates a non-positive number: exactly 0 or 1 only at the far tails |
| `logit(p)` | ln(p / (1 − p)) | `log(p) - log1m(p)`; NaN outside [0, 1] |
| `log_inv_logit(x)` | ln inv_logit(x) | `-log1p_exp(-x)`: finite for very negative x |
| `log1m_inv_logit(x)` | ln(1 − inv_logit(x)) | `-log1p_exp(x)`: finite for very positive x |
| `tanh(x)` | hyperbolic tangent | |

## Elementwise, two arguments: a real broadcasts

| Function | Value | Notes |
| --- | --- | --- |
| `fmin(x, y)` | smaller | a NaN argument is ignored (C `fmin`) |
| `fmax(x, y)` | larger | a NaN argument is ignored |
| `log_diff_exp(a, b)` | ln(eᵃ − eᵇ), a ≥ b | `a + log1m_exp(b - a)`; −inf when a == b, NaN when b > a |

## Reductions: a vector (or two reals) to a real

| Function | Value | Notes |
| --- | --- | --- |
| `sum(x)` | Σ xᵢ | |
| `dot_product(a, b)` | Σ aᵢbᵢ | sizes must agree |
| `log_sum_exp(x)`, `log_sum_exp(a, b)` | ln Σ e^xᵢ | shifted by the maximum, never overflows; −inf for empty or all −inf |
| `max(x)`, `max(a, b)` | largest | −inf for empty; NaN if any element is NaN |
| `min(x)`, `min(a, b)` | smallest | +inf for empty; NaN if any element is NaN |
| `mean(x)` | Σ xᵢ / n | NaN for empty |
| `dot_self(x)` | Σ xᵢ² | |
| `norm2(x)` | √Σ xᵢ² | scaled by max\|xᵢ\|: no overflow or underflow |
| `squared_distance(a, b)` | Σ (aᵢ − bᵢ)² | sizes must agree |

## Vector to vector

| Function | Value | Notes |
| --- | --- | --- |
| `rep_vector(x, n)` | n copies of x | |
| `softmax(x)` | eˣ / Σ eˣ | `exp(x - max(x)) / sum`: never overflows |
| `log_softmax(x)` | x − log_sum_exp(x) | finite where `softmax` underflows to 0 |

## Matrices

There are no arrays and no slices (`m[:, j]`) in a kernel. A set of K
vectors is a `matrix[dim, K]`, one vector per column, and these functions
move whole rows and columns in and out. Indexing is 1-based, and `m[i, j]`
still reads or writes one element.

A function cannot change its argument, so `set_col` and `set_row` return a
new matrix. Assign it back:

```lapkernel
matrix[dim, K] y;
for (j in 1:K) {
  y = set_col(y, j, q + s .* std_normal_rng(dim));
}
vector[dim] y2 = col(y, 2);
```

| Function | Value | Notes |
| --- | --- | --- |
| `rep_matrix(x, r, c)` | `matrix[r, c]`, every element x | |
| `rows(m)`, `cols(m)` | `int` | on a vector: its size, and 1 |
| `col(m, j)` | column j, a `vector` of `rows(m)` elements | |
| `row(m, i)` | row i, a `vector` of `cols(m)` elements | there is no `row_vector` type |
| `set_col(m, j, v)` | m with column j replaced by v | v has `rows(m)` elements |
| `set_row(m, i, v)` | m with row i replaced by v | v has `cols(m)` elements |

A wrong-length `v`, or an index that is a constant and out of range, is an
error when the kernel is checked. Any other index is checked when it runs.
Each `set_` call copies the matrix: filling all K columns of a
`matrix[dim, K]` costs O(dim·K²), which is negligible at the sizes a kernel
works with.

## Checks and constants

| Function | Value |
| --- | --- |
| `is_inf(x)` | `int`: 1 if x is ±inf |
| `is_nan(x)` | `int`: 1 if x is NaN |
| `pi()`, `e()` | π, e |
| `negative_infinity()`, `positive_infinity()` | ∓inf. `propose(q_new, negative_infinity())` rejects on purpose |

## Your own function with a library name

A library function (pure, allowed in tuning defaults) is **not reserved**. A
kernel may define a function or variable with the same name, and inside
that kernel its own definition wins. This means the library can grow
without breaking published kernels: laplace-langevin 0.1.0's `barker`
defines its own `log1p_exp` and still checks. Context names (`q`,
`log_density`, ...), `propose`/`move` and the RNGs stay reserved.

## Candidates, not yet added

- **More random numbers:** `exponential_rng`, `bernoulli_rng`,
  `uniform_rng(vector, vector)`. Each one fixes how many draws it takes from
  the chain stream, which the equivalence tests replay, so each needs a
  documented draw order.
- **Special functions:** `Phi`, `Phi_approx`, `std_normal_lcdf`, `lgamma`,
  `erf`. These need an implementation of their own (Rust's `std` has none).
- **Trigonometry:** `sin`, `cos`, `atan2`.
- **Statistics:** `variance`, `sd`, `cumulative_sum`.
- From the original list: Hessian, Hessian-vector product, `is_warmup`,
  `iteration` (`docs/sampler/open-questions.md`).
