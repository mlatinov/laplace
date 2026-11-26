# Kernel function library

Every function a `.lapkernel` can call without defining it. A kernel step
runs in laplace-sample's interpreter, not as compiled Stan, so a kernel
gets **this list, not Stan's whole library**. The source of truth is the
registry in `laplace-sampler-syntax/src/lapkernel/builtins.rs` and
`builtins/stdlib.rs` (names, type rules, docs) and
`laplace-sample/src/lapkernel/natives.rs` and `natives/stdlib.rs` (what each
one computes). A test checks that the two lists agree, and the deterministic
functions are checked against SciPy to 1e-12. Editors show the same text on
hover.

Names and meanings follow Stan, so a Stan user can guess them. Where Stan
has a numerically safe form of a common expression, only the safe form is
offered: write `log1p_exp(x)`, not `log(1 + exp(x))`.

**Elementwise** means that `f(vector)` and `f(matrix)` apply `f` to each
element and return the same shape. In two-argument elementwise functions a
`real` broadcasts against a `vector`. `int` arguments are promoted to
`real` everywhere.

**Out of domain** gives NaN (IEEE behaviour), not an error. A NaN that
reaches `propose()` rejects the proposal. A NaN position given to `move()`
stops the run.

**Densities** (`_lpdf`, `_lpmf`, `_lcdf`, `_lccdf`) are written
`f(y | ...)`, as in Stan, include their normalizing constants, and sum over
the elements when an argument is a vector (a real broadcasts). An outcome
outside the support gives `-inf`. An invalid parameter (a scale ≤ 0, a
probability outside [0, 1], a covariance that is not positive definite) is
an error that stops the run, naming the function and the value.

**Random numbers** broadcast the same way: `gamma_rng(alpha, beta)` with a
vector `alpha` gives a vector of draws. Stan returns an array there; a kernel
has no arrays, so it gets a vector. Each function documents how many numbers
it takes from the chain's stream (`laplace sampler doc`, editor hover).
Those that use a `rand_distr` rejection sampler (`gamma_rng`, `beta_rng`,
`student_t_rng`, `binomial_rng`, `poisson_rng`, `dirichlet_rng`) take a
variable number, so a kernel using them reproduces only with the same
laplace-sample version.

**Rounding returns a `real`**, as in Stan: `floor(2.7)` is `2.0`. Convert
with `to_int(floor(x))` where an `int` is needed (an index, a loop bound).

## Context, randomness, densities, ending a step

These are unchanged. See `docs/sampler/wiki/Level-3-Write-kernels.md`.

| Name | What |
| --- | --- |
| `q`, `lp`, `grad`, `dim`, `metric` | chain state and the adapted metric |
| `log_density(x)`, `log_density_grad(x)`, `leapfrog(q, p, eps, steps, m)` | the target |
| `std_normal_rng()`, `std_normal_rng(n)`, `normal_rng(mu, sigma)`, `uniform_rng(a, b)` | random numbers |
| `normal_lpdf(y \| mu, sigma)`, `std_normal_lpdf(y)` | log densities |
| `propose(q_new, log_correction)`, `move(q_new)` | end the step |

## Elementwise, one argument: `real f(real)`, `vector f(vector)`, `matrix f(matrix)`

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
| `sin`, `cos`, `tan` | trigonometric (radians) | |
| `asin`, `acos`, `atan` | inverse trigonometric | NaN outside [−1, 1] for `asin`, `acos` |
| `sinh`, `cosh`, `asinh`, `acosh`, `atanh` | hyperbolic and inverses | `acosh` NaN below 1; `atanh` ±inf at ±1 |
| `log2(x)`, `log10(x)` | base-2, base-10 logarithm | |
| `exp2(x)` | 2ˣ | |
| `cbrt(x)` | ∛x | negative for negative x |
| `floor`, `ceil`, `round`, `trunc` | rounding, **as a real** | `round` takes halves away from zero |
| `erf(x)`, `erfc(x)` | error function, 1 − erf(x) | `erfc` accurate in the tail (libm) |
| `Phi(x)` | standard normal CDF | `0.5 * erfc(-x / √2)`: exactly 0 only below about −38 |
| `Phi_approx(x)` | logistic approximation of Phi | `inv_logit(0.07056 x³ + 1.5976 x)`, as Stan |
| `inv_Phi(p)` | standard normal quantile | Acklam + one Halley step: about 1e-15 relative; ∓inf at 0 and 1 |
| `lgamma(x)` | ln \|Γ(x)\| | |
| `tgamma(x)` | Γ(x) | overflows above about 171.6 |
| `digamma(x)` | ψ(x) | NaN at 0 and negative integers |
| `inv_cloglog(x)` | 1 − exp(−eˣ) | `-expm1(-exp(x))` |

## Elementwise, two arguments: a real broadcasts

| Function | Value | Notes |
| --- | --- | --- |
| `fmin(x, y)` | smaller | a NaN argument is ignored (C `fmin`) |
| `fmax(x, y)` | larger | a NaN argument is ignored |
| `log_diff_exp(a, b)` | ln(eᵃ − eᵇ), a ≥ b | `a + log1m_exp(b - a)`; −inf when a == b, NaN when b > a |
| `pow(x, y)` | xʸ | as `^` |
| `atan2(y, x)` | angle of (x, y) | in (−π, π] |
| `hypot(x, y)` | √(x² + y²) | no overflow |
| `fdim(x, y)` | max(x − y, 0) | |
| `fmod(x, y)` | remainder of x / y | sign of x |
| `lbeta(a, b)` | ln B(a, b) | through `lgamma` |
| `lchoose(n, k)` | ln of the binomial coefficient | through `lgamma`: real n and k allowed |
| `log_mix(theta, lp1, lp2)` | ln(θ e^lp1 + (1 − θ) e^lp2) | reals only; no overflow |

## Reductions: a vector or matrix (or two reals) to a real

`sum`, `mean`, `max`, `min`, `log_sum_exp`, `prod`, `variance` and `sd` take
a matrix as well, over all its elements.

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
| `distance(a, b)` | √Σ (aᵢ − bᵢ)² | sizes must agree |
| `norm1(x)` | Σ \|xᵢ\| | |
| `prod(x)` | Π xᵢ | 1 for empty |
| `variance(x)`, `sd(x)` | sample variance (n − 1), its square root | an error for fewer than 2 elements |

## Building and reshaping vectors

| Function | Value | Notes |
| --- | --- | --- |
| `rep_vector(x, n)` | n copies of x | |
| `zeros_vector(n)`, `ones_vector(n)` | n zeros, n ones | |
| `linspaced_vector(n, lo, hi)` | n evenly spaced values, lo and hi included | |
| `one_hot_vector(n, k)` | zeros with a 1 at position k | |
| `head(v, n)`, `tail(v, n)` | first, last n elements | |
| `segment(v, i, n)` | n elements from position i | |
| `append_row(a, b)` | a stacked on b | vector/real with vector, or matrix with matrix (same columns) |
| `reverse(v)`, `sort_asc(v)`, `sort_desc(v)` | reordered | sorting a NaN is an error |
| `cumulative_sum(v)` | running sums | |
| `softmax(x)` | eˣ / Σ eˣ | `exp(x - max(x)) / sum`: never overflows |
| `log_softmax(x)` | x − log_sum_exp(x) | finite where `softmax` underflows to 0 |

## Integers and conversions

| Function | Value | Notes |
| --- | --- | --- |
| `to_int(x)` | `int`: x truncated toward zero | an error for NaN, ±inf or out of the 64-bit range |
| `int_step(x)` | `int`: 1 if x > 0, else 0 | |
| `choose(n, k)` | `int`: binomial coefficient | exact; an error on overflow |
| `size(x)`, `num_elements(x)` | `int`: number of elements | rows · cols for a matrix |
| `to_vector(m)` | the elements of m, column-major | as Stan; also a vector or a real |
| `to_matrix(v, r, c)` | v filled column-major into r × c | `to_matrix(v)` is one column; a matrix is reshaped the same way |

`min`, `max` and `abs` of `int` arguments still return a `real` (as they did
in 0.3.0, so `min(a, b) / 2` keeps meaning real division). Wrap in `to_int`
for an `int`.

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
| `append_col(a, b)` | b to the right of a | same rows; a vector is one column |
| `to_vector(m)`, `to_matrix(v, r, c)` | flatten, reshape | column-major |

A wrong-length `v`, or an index that is a constant and out of range, is an
error when the kernel is checked. Any other index is checked when it runs.
Each `set_` call copies the matrix: filling all K columns of a
`matrix[dim, K]` costs O(dim·K²), which is negligible at the sizes a kernel
works with.

`*` already multiplies matrix × vector and matrix × matrix. There is no
transpose operator (`'`) in a kernel; `crossprod`, `tcrossprod` and the
`mdivide_` functions cover the usual uses.

## Linear algebra

| Function | Value | Notes |
| --- | --- | --- |
| `identity_matrix(n)` | n × n identity | |
| `diag_matrix(v)`, `diagonal(m)` | v on a diagonal; the diagonal of m | |
| `trace(m)` | Σ mᵢᵢ | |
| `determinant(m)`, `log_determinant(m)` | det, ln \|det\| | through LU; `log_determinant` never overflows |
| `inverse(m)` | m⁻¹ | an error if singular |
| `cholesky_decompose(m)` | lower-triangular L, L L' = m | an error unless symmetric (to 1e-8) and positive definite |
| `eigenvalues_sym(m)`, `eigenvectors_sym(m)` | eigen-decomposition of a symmetric m | ascending, as Stan; vectors as columns |
| `crossprod(m)`, `tcrossprod(m)` | m' m, m m' | |
| `multiply_lower_tri_self_transpose(L)` | L L' | reads the lower triangle only |
| `quad_form(A, v)` | v' A v | |
| `mdivide_left(A, b)` | A⁻¹ b | through LU; b a vector or matrix |
| `mdivide_left_tri_low(L, b)` | L⁻¹ b | forward substitution |
| `mdivide_left_spd(A, b)` | A⁻¹ b, A symmetric positive definite | through Cholesky |

## Random numbers

All draw from the chain's stream and are allowed only in `init { }` and
`step { }`.

| Function | Draws | Random numbers taken |
| --- | --- | --- |
| `std_normal_rng()`, `std_normal_rng(n)` | N(0, 1) | one normal per element |
| `normal_rng(mu, sigma)` | N(mu, sigma) | one normal per element |
| `uniform_rng(a, b)` | U[a, b) | one uniform per element |
| `exponential_rng(beta)` | rate beta | one `Exp1` per element |
| `gamma_rng(alpha, beta)` | shape alpha, **rate** beta (as Stan) | variable (`rand_distr::Gamma`) |
| `chi_square_rng(nu)` | χ²(nu) | variable |
| `beta_rng(a, b)` | Beta(a, b) | variable |
| `lognormal_rng(mu, sigma)` | exp(N(mu, sigma)) | one normal per element |
| `cauchy_rng(mu, sigma)` | Cauchy | one uniform per element |
| `double_exponential_rng(mu, sigma)` | Laplace | one uniform per element |
| `student_t_rng(nu, mu, sigma)` | Student-t | variable |
| `bernoulli_rng(theta)`, `bernoulli_logit_rng(alpha)` | `int` 0 or 1 | one uniform |
| `binomial_rng(N, theta)` | `int` | variable |
| `poisson_rng(lambda)`, `poisson_log_rng(alpha)` | `int` | variable; lambda < 2³⁰ |
| `categorical_rng(theta)` | `int` in 1..K, P(k) = θₖ | one uniform; θ must sum to 1 (to 1e-8) |
| `categorical_logit_rng(beta)` | `int` in 1..K, P(k) = softmax(β)ₖ | one uniform; no need to normalize |
| `discrete_range_rng(lo, hi)` | `int` uniform on lo..hi | one uniform |
| `multi_normal_rng(mu, Sigma)` | N(mu, Σ) | size(mu) normals |
| `multi_normal_cholesky_rng(mu, L)` | N(mu, L L') | size(mu) normals |
| `dirichlet_rng(alpha)` | a simplex | variable |

## Densities, mass functions and log CDFs

Written `f(y | ...)`. Real arguments may be vectors (summed); `int`
outcomes are single values.

| Function | Notes |
| --- | --- |
| `normal_lpdf`, `std_normal_lpdf` | |
| `exponential_lpdf(y \| beta)` | rate beta |
| `gamma_lpdf(y \| alpha, beta)` | shape, rate |
| `beta_lpdf(y \| a, b)` | |
| `lognormal_lpdf(y \| mu, sigma)` | |
| `cauchy_lpdf(y \| mu, sigma)` | |
| `double_exponential_lpdf(y \| mu, sigma)` | Laplace |
| `logistic_lpdf(y \| mu, sigma)` | |
| `student_t_lpdf(y \| nu, mu, sigma)` | |
| `uniform_lpdf(y \| a, b)` | an error unless a < b |
| `multi_normal_lpdf(y \| mu, Sigma)`, `multi_normal_cholesky_lpdf(y \| mu, L)` | vector y |
| `normal_lcdf(y \| mu, sigma)`, `normal_lccdf(y \| mu, sigma)` | accurate far into the tails |
| `std_normal_lcdf(y)`, `std_normal_lccdf(y)` | |
| `bernoulli_lpmf(y \| theta)`, `bernoulli_logit_lpmf(y \| alpha)` | |
| `binomial_lpmf(n \| N, theta)` | |
| `poisson_lpmf(n \| lambda)`, `poisson_log_lpmf(n \| alpha)` | |
| `categorical_lpmf(k \| theta)`, `categorical_logit_lpmf(k \| beta)` | |

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
`log_density`, ...), `propose`/`move` and the RNGs stay reserved, so a new
`_rng` is the one kind of addition that can clash with a kernel's own
function of that name.

## Candidates, not yet added

- **A transpose operator** (`m'`) and a `row_vector` type: a grammar change,
  so not a library addition. `crossprod`, `tcrossprod` and `mdivide_` cover
  most uses.
- **Arrays** (`array[K] vector[dim]`): also a grammar change. Use the
  columns of a matrix.
- **Integer `min`/`max`/`abs`**: they return `real` today, and changing
  that would change what existing kernels compute.
- **More distributions** as needed: `weibull`, `frechet`, `inv_gamma`,
  `wishart`, `lkj_corr_cholesky`, `neg_binomial`. Each is one row in each
  stdlib file.
- From the original list: Hessian, Hessian-vector product, `is_warmup`,
  `iteration` (`docs/sampler/open-questions.md`).
