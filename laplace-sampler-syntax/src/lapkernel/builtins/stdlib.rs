//! The kernel standard library beyond v1: Stan's names and meanings, for
//! the types a kernel has (`int`, `real`, `vector`, `matrix`). One row per
//! function; the implementations are in laplace-sample
//! (`lapkernel::natives::stdlib`), keyed by the same names.
//!
//! Conventions, shared by every row (`docs/sampler/kernel-functions.md`):
//! - **Elementwise** functions take a real, a vector or a matrix and return
//!   the same shape; `int` promotes to `real`.
//! - **Random numbers** broadcast like `normal_rng`: a vector argument
//!   gives a vector of draws, a real broadcasts. Stan returns arrays there;
//!   a kernel has none, so it gets a vector.
//! - **Densities** (`_lpdf`, `_lpmf`, `_lcdf`, `_lccdf`) are written
//!   `f(y | ...)`, include their constants, and sum over broadcast
//!   elements. An outcome outside the support gives `-inf`; an invalid
//!   parameter (a scale <= 0, a probability outside [0, 1]) is an error.
//! - Integer-valued rounding (`floor`, `round`, ...) returns a `real`, as in
//!   Stan; `to_int` converts.

use super::{
    bad_args, binary_math_sig, broadcast_type, const_size, reduce_sig, unary_math_sig,
    vector_map_sig, Entry, Kind, SigFn,
};
use crate::lapkernel::types::{Size, Ty};

/// Name, signatures, documentation, RNG use ("none" for none), type rule.
type Row = (
    &'static str,
    &'static [&'static str],
    &'static str,
    &'static str,
    SigFn,
);

/// `f(real x)`, `f(vector x)`, `f(matrix x)`: same shape out.
macro_rules! elementwise {
    ($name:literal, $doc:expr) => {
        (
            $name,
            &[
                concat!("real ", $name, "(real x)"),
                concat!("vector ", $name, "(vector x)"),
                concat!("matrix ", $name, "(matrix x)"),
            ],
            $doc,
            NONE,
            unary_math_sig as SigFn,
        )
    };
}

/// `f(real x, real y)`; either may be a vector, and a real broadcasts.
macro_rules! elementwise2 {
    ($name:literal, $x:literal, $y:literal, $doc:expr) => {
        (
            $name,
            &[
                concat!("real ", $name, "(real ", $x, ", real ", $y, ")"),
                concat!("vector ", $name, "(real|vector ", $x, ", vector ", $y, ")"),
                concat!("vector ", $name, "(vector ", $x, ", real ", $y, ")"),
            ],
            $doc,
            NONE,
            binary_math_sig as SigFn,
        )
    };
}

const NONE: &str = "none";

pub(super) fn push(entries: &mut Vec<Entry>) {
    let rows: &[Row] = &[
        // ---- trigonometric and hyperbolic ------------------------------------
        elementwise!("sin", "sine (x in radians), elementwise"),
        elementwise!("cos", "cosine (x in radians), elementwise"),
        elementwise!("tan", "tangent (x in radians), elementwise"),
        elementwise!("asin", "arcsine, in [-π/2, π/2], elementwise. NaN outside [-1, 1]"),
        elementwise!("acos", "arccosine, in [0, π], elementwise. NaN outside [-1, 1]"),
        elementwise!("atan", "arctangent, in (-π/2, π/2), elementwise"),
        elementwise!("sinh", "hyperbolic sine, elementwise"),
        elementwise!("cosh", "hyperbolic cosine, elementwise"),
        elementwise!("asinh", "inverse hyperbolic sine, elementwise"),
        elementwise!("acosh", "inverse hyperbolic cosine, elementwise. NaN below 1"),
        elementwise!("atanh", "inverse hyperbolic tangent, elementwise. ±inf at ±1, NaN outside"),
        // ---- powers and logarithms -------------------------------------------
        elementwise!("log2", "base-2 logarithm, elementwise"),
        elementwise!("log10", "base-10 logarithm, elementwise"),
        elementwise!("exp2", "2^x, elementwise"),
        elementwise!("cbrt", "cube root (negative for negative x), elementwise"),
        // ---- rounding (a real result, as in Stan; to_int converts) -----------
        elementwise!("floor", "largest integer <= x, as a real, elementwise"),
        elementwise!("ceil", "smallest integer >= x, as a real, elementwise"),
        elementwise!("round", "nearest integer, halves away from zero, as a real, elementwise"),
        elementwise!("trunc", "x rounded toward zero, as a real, elementwise"),
        // ---- special functions -----------------------------------------------
        elementwise!("erf", "error function, elementwise"),
        elementwise!("erfc", "complementary error function 1 - erf(x), accurate for large x, elementwise"),
        elementwise!("Phi", "standard normal CDF, 0.5 * erfc(-x / √2): exactly 0 only below about -38, elementwise"),
        elementwise!("Phi_approx", "logistic approximation of Phi: inv_logit(0.07056 x³ + 1.5976 x), elementwise"),
        elementwise!("inv_Phi", "standard normal quantile (Acklam's approximation plus one Newton step, \
                     about 1e-15 relative), elementwise. -inf at 0, +inf at 1, NaN outside [0, 1]"),
        elementwise!("lgamma", "log |Γ(x)|, elementwise. +inf at 0 and negative integers"),
        elementwise!("tgamma", "Γ(x), elementwise. Overflows to +inf above about 171.6"),
        elementwise!("digamma", "ψ(x) = d/dx log Γ(x), elementwise. NaN at 0 and negative integers"),
        elementwise!("inv_cloglog", "1 - exp(-exp(x)), computed as -expm1(-exp(x)), elementwise"),
        // ---- two arguments ---------------------------------------------------
        elementwise2!("pow", "x", "y", "x^y, elementwise (the same as the `^` operator on reals)"),
        elementwise2!("atan2", "y", "x", "angle of the point (x, y), in (-π, π], elementwise"),
        elementwise2!("hypot", "x", "y", "sqrt(x² + y²) without overflow, elementwise"),
        elementwise2!("fdim", "x", "y", "positive difference max(x - y, 0), elementwise"),
        elementwise2!("fmod", "x", "y", "remainder of x / y, with the sign of x, elementwise"),
        elementwise2!("lbeta", "a", "b", "log B(a, b) = lgamma(a) + lgamma(b) - lgamma(a + b), elementwise"),
        elementwise2!("lchoose", "n", "k", "log of the binomial coefficient, through lgamma: real n and k allowed, elementwise"),
        ("log_mix", &["real log_mix(real theta, real lp1, real lp2)"],
            "log(theta * exp(lp1) + (1 - theta) * exp(lp2)), without overflow. \
             theta outside [0, 1] is an error", NONE, log_mix_sig),
        // ---- integers and conversions ----------------------------------------
        ("to_int", &["int to_int(real x)"],
            "x truncated toward zero, as an int. An error for NaN, ±inf, or a value \
             outside the 64-bit integer range", NONE, real_to_int_sig),
        ("int_step", &["int int_step(real x)"], "1 if x > 0, else 0 (also 0 for NaN)", NONE, real_to_int_sig),
        ("choose", &["int choose(int n, int k)"],
            "binomial coefficient n over k, exact: 0 if k > n, an error if n or k is \
             negative or the result overflows", NONE, choose_sig),
        ("to_vector", &["vector to_vector(matrix m)", "vector to_vector(vector v)", "vector to_vector(real x)"],
            "the elements of m in column-major order (as Stan); a vector unchanged; a real as a \
             vector of one", NONE, to_vector_sig),
        ("to_matrix", &["matrix[r, c] to_matrix(vector v, int r, int c)", "matrix to_matrix(vector v)",
                        "matrix to_matrix(matrix m, int r, int c)"],
            "v filled into r rows and c columns in column-major order (as Stan); one column if no \
             sizes are given. A matrix is reshaped the same way. r * c must equal the number of elements",
            NONE, to_matrix_sig),
        ("size", &["int size(vector v)", "int size(matrix m)"],
            "number of elements (rows * cols for a matrix)", NONE, size_sig),
        ("num_elements", &["int num_elements(vector v)", "int num_elements(matrix m)"],
            "number of elements (rows * cols for a matrix)", NONE, size_sig),
        // ---- reductions --------------------------------------------------------
        ("prod", &["real prod(vector x)", "real prod(matrix x)"], "product of the elements (1 if empty)", NONE, reduce_sig),
        ("variance", &["real variance(vector x)", "real variance(matrix x)"],
            "sample variance, divided by n - 1 (as Stan). An error for fewer than 2 elements", NONE, reduce_sig),
        ("sd", &["real sd(vector x)", "real sd(matrix x)"],
            "sample standard deviation, sqrt(variance(x)). An error for fewer than 2 elements", NONE, reduce_sig),
        ("norm1", &["real norm1(vector x)"], "sum of |x[i]|", NONE, reduce_sig),
        ("distance", &["real distance(vector a, vector b)"], "Euclidean distance, sqrt(squared_distance(a, b))", NONE, two_vectors_real_sig),
        // ---- building and reshaping vectors --------------------------------------
        ("zeros_vector", &["vector[n] zeros_vector(int n)"], "n zeros", NONE, sized_vector_sig),
        ("ones_vector", &["vector[n] ones_vector(int n)"], "n ones", NONE, sized_vector_sig),
        ("linspaced_vector", &["vector[n] linspaced_vector(int n, real lo, real hi)"],
            "n evenly spaced values from lo to hi, both included (hi alone if n = 1)", NONE, linspaced_sig),
        ("one_hot_vector", &["vector[n] one_hot_vector(int n, int k)"],
            "n zeros with a 1 at position k (1 <= k <= n)", NONE, one_hot_sig),
        ("head", &["vector[n] head(vector v, int n)"], "the first n elements of v", NONE, head_tail_sig),
        ("tail", &["vector[n] tail(vector v, int n)"], "the last n elements of v", NONE, head_tail_sig),
        ("segment", &["vector[n] segment(vector v, int i, int n)"],
            "n elements of v starting at position i", NONE, segment_sig),
        ("append_row", &["vector append_row(real|vector a, vector b)", "vector append_row(vector a, real b)",
                         "matrix append_row(matrix a, matrix b)"],
            "a stacked on top of b: vectors (or a real and a vector) into a longer vector, matrices \
             with the same number of columns into a taller matrix", NONE, append_row_sig),
        ("append_col", &["matrix append_col(matrix|vector a, matrix|vector b)"],
            "b to the right of a, as columns: the two need the same number of rows (a vector is one column)",
            NONE, append_col_sig),
        ("reverse", &["vector reverse(vector v)"], "the elements in reverse order", NONE, vector_map_sig),
        ("sort_asc", &["vector sort_asc(vector v)"], "the elements in ascending order (NaN is an error)", NONE, vector_map_sig),
        ("sort_desc", &["vector sort_desc(vector v)"], "the elements in descending order (NaN is an error)", NONE, vector_map_sig),
        ("cumulative_sum", &["vector cumulative_sum(vector v)"], "running sums: element i is v[1] + ... + v[i]", NONE, vector_map_sig),
        // ---- matrices and linear algebra -----------------------------------------
        ("identity_matrix", &["matrix[n, n] identity_matrix(int n)"], "the n x n identity", NONE, identity_sig),
        ("diag_matrix", &["matrix diag_matrix(vector v)"], "square matrix with v on the diagonal, zeros elsewhere", NONE, diag_matrix_sig),
        ("diagonal", &["vector diagonal(matrix m)"], "the diagonal of m (min(rows, cols) elements)", NONE, diagonal_sig),
        ("trace", &["real trace(matrix m)"], "sum of the diagonal", NONE, matrix_to_real_sig),
        ("determinant", &["real determinant(matrix m)"], "determinant of a square matrix (through LU)", NONE, square_to_real_sig),
        ("log_determinant", &["real log_determinant(matrix m)"],
            "log |det(m)| of a square matrix, without overflow (through LU)", NONE, square_to_real_sig),
        ("inverse", &["matrix inverse(matrix m)"], "inverse of a square matrix. An error if it is singular", NONE, square_sig),
        ("cholesky_decompose", &["matrix cholesky_decompose(matrix m)"],
            "lower-triangular L with L * L' = m. An error unless m is symmetric positive definite", NONE, square_sig),
        ("eigenvalues_sym", &["vector eigenvalues_sym(matrix m)"],
            "eigenvalues of a symmetric matrix, ascending (as Stan)", NONE, eigenvalues_sig),
        ("eigenvectors_sym", &["matrix eigenvectors_sym(matrix m)"],
            "eigenvectors of a symmetric matrix as columns, in the order of eigenvalues_sym", NONE, square_sig),
        ("crossprod", &["matrix crossprod(matrix m)"], "m' * m", NONE, crossprod_sig),
        ("tcrossprod", &["matrix tcrossprod(matrix m)"], "m * m'", NONE, tcrossprod_sig),
        ("multiply_lower_tri_self_transpose", &["matrix multiply_lower_tri_self_transpose(matrix L)"],
            "L * L', reading only the lower triangle of L (as Stan)", NONE, tcrossprod_sig),
        ("quad_form", &["real quad_form(matrix A, vector v)"], "v' * A * v", NONE, quad_form_sig),
        ("mdivide_left", &["vector mdivide_left(matrix A, vector b)", "matrix mdivide_left(matrix A, matrix B)"],
            "the solution x of A * x = b (through LU). An error if A is singular", NONE, mdivide_sig),
        ("mdivide_left_tri_low", &["vector mdivide_left_tri_low(matrix L, vector b)",
                                   "matrix mdivide_left_tri_low(matrix L, matrix B)"],
            "the solution x of L * x = b for lower-triangular L, by forward substitution (only the \
             lower triangle is read)", NONE, mdivide_sig),
        ("mdivide_left_spd", &["vector mdivide_left_spd(matrix A, vector b)", "matrix mdivide_left_spd(matrix A, matrix B)"],
            "the solution x of A * x = b for symmetric positive-definite A (through Cholesky)", NONE, mdivide_sig),
        // ---- random numbers: continuous, broadcasting ------------------------------
        ("exponential_rng", &["real exponential_rng(real beta)", "vector exponential_rng(vector beta)"],
            "exponential draws with rate beta (mean 1 / beta)", "one rand_distr::Exp1 draw per element, in index order", rng1_sig),
        ("gamma_rng", &["real gamma_rng(real alpha, real beta)", "vector gamma_rng(vector alpha, vector beta)"],
            "gamma draws with shape alpha and rate beta (as Stan: mean alpha / beta)",
            "rand_distr::Gamma per element, in index order: a variable number of normals and uniforms", rng2_sig),
        ("beta_rng", &["real beta_rng(real a, real b)", "vector beta_rng(vector a, vector b)"],
            "beta draws on (0, 1)", "rand_distr::Beta per element, in index order: a variable number of uniforms", rng2_sig),
        ("lognormal_rng", &["real lognormal_rng(real mu, real sigma)", "vector lognormal_rng(vector mu, vector sigma)"],
            "exp(mu + sigma * z) with z ~ N(0, 1)", "one normal draw (rand_distr::StandardNormal) per element, in index order", rng2_sig),
        ("cauchy_rng", &["real cauchy_rng(real mu, real sigma)", "vector cauchy_rng(vector mu, vector sigma)"],
            "Cauchy draws, location mu and scale sigma: mu + sigma * tan(π (u - 1/2))",
            "one uniform per element, in index order", rng2_sig),
        ("double_exponential_rng", &["real double_exponential_rng(real mu, real sigma)",
                                     "vector double_exponential_rng(vector mu, vector sigma)"],
            "Laplace (double exponential) draws, location mu and scale sigma, by inverting the CDF",
            "one uniform per element, in index order", rng2_sig),
        ("student_t_rng", &["real student_t_rng(real nu, real mu, real sigma)",
                            "vector student_t_rng(vector nu, vector mu, vector sigma)"],
            "Student-t draws with nu degrees of freedom: mu + sigma * t",
            "rand_distr::StudentT per element, in index order: a variable number of normals and uniforms", rng3_sig),
        ("chi_square_rng", &["real chi_square_rng(real nu)", "vector chi_square_rng(vector nu)"],
            "chi-square draws with nu degrees of freedom", "rand_distr::Gamma(nu / 2, 2) per element, in index order", rng1_sig),
        // ---- random numbers: discrete (one int) --------------------------------------
        ("bernoulli_rng", &["int bernoulli_rng(real theta)"], "1 with probability theta, else 0", "one uniform", bern_rng_sig),
        ("bernoulli_logit_rng", &["int bernoulli_logit_rng(real alpha)"],
            "1 with probability inv_logit(alpha), else 0", "one uniform", bern_rng_sig),
        ("binomial_rng", &["int binomial_rng(int N, real theta)"], "successes in N trials of probability theta",
            "rand_distr::Binomial: a variable number of uniforms", binomial_rng_sig),
        ("poisson_rng", &["int poisson_rng(real lambda)"], "Poisson draw with mean lambda (lambda < 2^30)",
            "rand_distr::Poisson: a variable number of uniforms", bern_rng_sig),
        ("poisson_log_rng", &["int poisson_log_rng(real alpha)"], "Poisson draw with mean exp(alpha)",
            "rand_distr::Poisson: a variable number of uniforms", bern_rng_sig),
        ("categorical_rng", &["int categorical_rng(vector theta)"],
            "k in 1..size(theta) with probability theta[k]. theta must be non-negative and sum to 1 \
             (within 1e-8)", "one uniform, inverted through the cumulative sum", categorical_rng_sig),
        ("categorical_logit_rng", &["int categorical_logit_rng(vector beta)"],
            "k with probability softmax(beta)[k]: no need to normalize",
            "one uniform, inverted through the cumulative sum", categorical_rng_sig),
        ("discrete_range_rng", &["int discrete_range_rng(int lower, int upper)"],
            "an integer uniform on lower..upper, both included", "one uniform", discrete_range_rng_sig),
        // ---- random numbers: multivariate --------------------------------------------
        ("multi_normal_rng", &["vector multi_normal_rng(vector mu, matrix Sigma)"],
            "a draw from N(mu, Sigma): mu + L * z with L = cholesky_decompose(Sigma)",
            "size(mu) normal draws (rand_distr::StandardNormal), in index order", multi_normal_sig),
        ("multi_normal_cholesky_rng", &["vector multi_normal_cholesky_rng(vector mu, matrix L)"],
            "a draw from N(mu, L * L'): mu + L * z (only the lower triangle of L is read)",
            "size(mu) normal draws (rand_distr::StandardNormal), in index order", multi_normal_sig),
        ("dirichlet_rng", &["vector dirichlet_rng(vector alpha)"],
            "a simplex: independent gamma(alpha[k], 1) draws, normalized",
            "rand_distr::Gamma per element, in index order", vector_map_sig),
        // ---- densities ----------------------------------------------------------------
        ("exponential_lpdf", &["real exponential_lpdf(real|vector y | real|vector beta)"],
            "log beta - beta * y, summed; -inf for y < 0", NONE, dens2_sig),
        ("gamma_lpdf", &["real gamma_lpdf(real|vector y | real|vector alpha, real|vector beta)"],
            "gamma (shape alpha, rate beta) log density, summed; -inf for y < 0", NONE, dens3_sig),
        ("beta_lpdf", &["real beta_lpdf(real|vector y | real|vector a, real|vector b)"],
            "beta log density, summed; -inf outside [0, 1]", NONE, dens3_sig),
        ("lognormal_lpdf", &["real lognormal_lpdf(real|vector y | real|vector mu, real|vector sigma)"],
            "lognormal log density, summed; -inf for y <= 0", NONE, dens3_sig),
        ("cauchy_lpdf", &["real cauchy_lpdf(real|vector y | real|vector mu, real|vector sigma)"],
            "Cauchy log density, summed", NONE, dens3_sig),
        ("double_exponential_lpdf", &["real double_exponential_lpdf(real|vector y | real|vector mu, real|vector sigma)"],
            "Laplace log density -log(2 sigma) - |y - mu| / sigma, summed", NONE, dens3_sig),
        ("student_t_lpdf", &["real student_t_lpdf(real|vector y | real|vector nu, real|vector mu, real|vector sigma)"],
            "Student-t log density, summed", NONE, dens4_sig),
        ("uniform_lpdf", &["real uniform_lpdf(real|vector y | real|vector a, real|vector b)"],
            "-log(b - a) inside [a, b], -inf outside, summed. An error unless a < b", NONE, dens3_sig),
        ("logistic_lpdf", &["real logistic_lpdf(real|vector y | real|vector mu, real|vector sigma)"],
            "logistic log density, summed", NONE, dens3_sig),
        ("multi_normal_lpdf", &["real multi_normal_lpdf(vector y | vector mu, matrix Sigma)"],
            "multivariate normal log density (through Cholesky). An error unless Sigma is symmetric \
             positive definite", NONE, multi_normal_lpdf_sig),
        ("multi_normal_cholesky_lpdf", &["real multi_normal_cholesky_lpdf(vector y | vector mu, matrix L)"],
            "multivariate normal log density with covariance L * L'", NONE, multi_normal_lpdf_sig),
        ("normal_lcdf", &["real normal_lcdf(real|vector y | real|vector mu, real|vector sigma)"],
            "log P(Y <= y) for Y ~ N(mu, sigma), summed: accurate far into the lower tail", NONE, dens3_sig),
        ("normal_lccdf", &["real normal_lccdf(real|vector y | real|vector mu, real|vector sigma)"],
            "log P(Y > y) for Y ~ N(mu, sigma), summed: accurate far into the upper tail", NONE, dens3_sig),
        ("std_normal_lcdf", &["real std_normal_lcdf(real|vector y)"], "log Phi(y), summed", NONE, dens1_sig),
        ("std_normal_lccdf", &["real std_normal_lccdf(real|vector y)"], "log(1 - Phi(y)), summed", NONE, dens1_sig),
        ("bernoulli_lpmf", &["real bernoulli_lpmf(int y | real theta)"],
            "log theta for y = 1, log(1 - theta) for y = 0, -inf otherwise", NONE, int_pmf_sig),
        ("bernoulli_logit_lpmf", &["real bernoulli_logit_lpmf(int y | real alpha)"],
            "bernoulli_lpmf(y | inv_logit(alpha)), without overflow", NONE, int_pmf_sig),
        ("poisson_lpmf", &["real poisson_lpmf(int n | real lambda)"],
            "n log lambda - lambda - lgamma(n + 1); -inf for n < 0", NONE, int_pmf_sig),
        ("poisson_log_lpmf", &["real poisson_log_lpmf(int n | real alpha)"],
            "poisson_lpmf(n | exp(alpha)), without overflow", NONE, int_pmf_sig),
        ("binomial_lpmf", &["real binomial_lpmf(int n | int N, real theta)"],
            "binomial log probability; -inf unless 0 <= n <= N", NONE, binomial_lpmf_sig),
        ("categorical_lpmf", &["real categorical_lpmf(int k | vector theta)"],
            "log theta[k]; -inf for k outside 1..size(theta)", NONE, categorical_lpmf_sig),
        ("categorical_logit_lpmf", &["real categorical_logit_lpmf(int k | vector beta)"],
            "beta[k] - log_sum_exp(beta); -inf for k outside 1..size(beta)", NONE, categorical_lpmf_sig),
    ];
    for &(name, signatures, doc, rng, sig) in rows {
        entries.push(Entry {
            name,
            signatures,
            doc,
            rng,
            needs_chain: rng != NONE,
            kind: Kind::Function {
                sig,
                terminal: false,
            },
        });
    }
}

// ---- type rules ---------------------------------------------------------------------

type Consts<'a> = &'a [Option<i64>];

fn scalar_or_vector(t: &Ty) -> bool {
    t.is_scalar() || matches!(t, Ty::Vector(_))
}

/// `n` scalar-or-vector arguments of agreeing sizes: their broadcast type.
fn broadcast_n(a: &[Ty], n: usize, sig: &str) -> Result<Ty, String> {
    if a.len() != n || !a.iter().all(scalar_or_vector) {
        return Err(bad_args(sig, a));
    }
    broadcast_type(a)
}

fn add_sizes(a: Size, b: Size) -> Size {
    match (a, b) {
        (Size::Known(x), Size::Known(y)) => Size::Known(x + y),
        _ => Size::Unknown,
    }
}

fn mul_sizes(a: Size, b: Size) -> Size {
    match (a, b) {
        (Size::Known(x), Size::Known(y)) => Size::Known(x * y),
        _ => Size::Unknown,
    }
}

/// The sizes of a square matrix, or an error naming the function's shape.
fn square(r: Size, c: Size) -> Result<Size, String> {
    if r.agrees(c) {
        Ok(r.meet(c))
    } else {
        Err(format!("needs a square matrix, got {r} x {c}"))
    }
}

fn log_mix_sig(a: &[Ty], _: Consts, _: Option<usize>) -> Result<Ty, String> {
    match a {
        [x, y, z] if x.is_scalar() && y.is_scalar() && z.is_scalar() => Ok(Ty::Real),
        _ => Err(bad_args("log_mix(real theta, real lp1, real lp2)", a)),
    }
}

fn real_to_int_sig(a: &[Ty], _: Consts, _: Option<usize>) -> Result<Ty, String> {
    match a {
        [t] if t.is_scalar() => Ok(Ty::Int),
        _ => Err(bad_args("f(real x)", a)),
    }
}

fn choose_sig(a: &[Ty], _: Consts, _: Option<usize>) -> Result<Ty, String> {
    match a {
        [Ty::Int, Ty::Int] => Ok(Ty::Int),
        _ => Err(bad_args("choose(int n, int k)", a)),
    }
}

fn to_vector_sig(a: &[Ty], _: Consts, _: Option<usize>) -> Result<Ty, String> {
    match a {
        [Ty::Matrix(r, c)] => Ok(Ty::Vector(mul_sizes(*r, *c))),
        [Ty::Vector(n)] => Ok(Ty::Vector(*n)),
        [t] if t.is_scalar() => Ok(Ty::Vector(Size::Known(1))),
        _ => Err(bad_args(
            "to_vector(matrix m), to_vector(vector v) or to_vector(real x)",
            a,
        )),
    }
}

fn to_matrix_sig(a: &[Ty], k: Consts, _: Option<usize>) -> Result<Ty, String> {
    let count = match a {
        [Ty::Vector(n)] => return Ok(Ty::Matrix(*n, Size::Known(1))),
        [Ty::Vector(n), Ty::Int, Ty::Int] => *n,
        [Ty::Matrix(r, c), Ty::Int, Ty::Int] => mul_sizes(*r, *c),
        _ => {
            return Err(bad_args(
                "to_matrix(vector v), to_matrix(vector v, int r, int c) or to_matrix(matrix m, int r, int c)",
                a,
            ))
        }
    };
    let (r, c) = (const_size(k[1])?, const_size(k[2])?);
    if let (Size::Known(n), Size::Known(r), Size::Known(c)) = (count, r, c) {
        if r * c != n {
            return Err(format!(
                "{r} x {c} is {} elements, the input has {n}",
                r * c
            ));
        }
    }
    Ok(Ty::Matrix(r, c))
}

fn size_sig(a: &[Ty], _: Consts, _: Option<usize>) -> Result<Ty, String> {
    match a {
        [Ty::Vector(_) | Ty::Matrix(..)] => Ok(Ty::Int),
        _ => Err(bad_args("f(vector v) or f(matrix m)", a)),
    }
}

fn two_vectors_real_sig(a: &[Ty], _: Consts, _: Option<usize>) -> Result<Ty, String> {
    match a {
        [Ty::Vector(x), Ty::Vector(y)] if x.agrees(*y) => Ok(Ty::Real),
        [Ty::Vector(x), Ty::Vector(y)] => Err(format!("vector sizes {x} and {y} differ")),
        _ => Err(bad_args("f(vector a, vector b)", a)),
    }
}

fn sized_vector_sig(a: &[Ty], k: Consts, _: Option<usize>) -> Result<Ty, String> {
    match a {
        [Ty::Int] => Ok(Ty::Vector(const_size(k[0])?)),
        _ => Err(bad_args("f(int n)", a)),
    }
}

fn linspaced_sig(a: &[Ty], k: Consts, _: Option<usize>) -> Result<Ty, String> {
    match a {
        [Ty::Int, x, y] if x.is_scalar() && y.is_scalar() => Ok(Ty::Vector(const_size(k[0])?)),
        _ => Err(bad_args("linspaced_vector(int n, real lo, real hi)", a)),
    }
}

fn one_hot_sig(a: &[Ty], k: Consts, _: Option<usize>) -> Result<Ty, String> {
    match a {
        [Ty::Int, Ty::Int] => {
            if let (Some(n), Some(i)) = (k[0], k[1]) {
                if i < 1 || i > n {
                    return Err(format!("position {i} is out of range 1..{n}"));
                }
            }
            Ok(Ty::Vector(const_size(k[0])?))
        }
        _ => Err(bad_args("one_hot_vector(int n, int k)", a)),
    }
}

fn head_tail_sig(a: &[Ty], k: Consts, _: Option<usize>) -> Result<Ty, String> {
    match a {
        [Ty::Vector(len), Ty::Int] => {
            let n = const_size(k[1])?;
            if let (Size::Known(n), Size::Known(len)) = (n, *len) {
                if n > len {
                    return Err(format!("{n} elements asked for, the vector has {len}"));
                }
            }
            Ok(Ty::Vector(n))
        }
        _ => Err(bad_args("f(vector v, int n)", a)),
    }
}

fn segment_sig(a: &[Ty], k: Consts, _: Option<usize>) -> Result<Ty, String> {
    match a {
        [Ty::Vector(len), Ty::Int, Ty::Int] => {
            let n = const_size(k[2])?;
            if let (Some(i), Size::Known(n), Size::Known(len)) = (k[1], n, *len) {
                if i < 1 || (i as usize) + n > len + 1 {
                    return Err(format!(
                        "{n} elements from position {i} do not fit in a vector of {len}"
                    ));
                }
            }
            Ok(Ty::Vector(n))
        }
        _ => Err(bad_args("segment(vector v, int i, int n)", a)),
    }
}

fn append_row_sig(a: &[Ty], _: Consts, _: Option<usize>) -> Result<Ty, String> {
    let len = |t: &Ty| match t {
        Ty::Vector(n) => Some(*n),
        t if t.is_scalar() => Some(Size::Known(1)),
        _ => None,
    };
    match a {
        [Ty::Matrix(r1, c1), Ty::Matrix(r2, c2)] => {
            if !c1.agrees(*c2) {
                return Err(format!("the matrices have {c1} and {c2} columns"));
            }
            Ok(Ty::Matrix(add_sizes(*r1, *r2), c1.meet(*c2)))
        }
        [x, y] if !(x.is_scalar() && y.is_scalar()) => match (len(x), len(y)) {
            (Some(n), Some(m)) => Ok(Ty::Vector(add_sizes(n, m))),
            _ => Err(bad_args(
                "append_row(real|vector a, real|vector b) or append_row(matrix a, matrix b)",
                a,
            )),
        },
        _ => Err(bad_args(
            "append_row(real|vector a, real|vector b) or append_row(matrix a, matrix b)",
            a,
        )),
    }
}

fn append_col_sig(a: &[Ty], _: Consts, _: Option<usize>) -> Result<Ty, String> {
    let shape = |t: &Ty| match t {
        Ty::Matrix(r, c) => Some((*r, *c)),
        Ty::Vector(n) => Some((*n, Size::Known(1))),
        _ => None,
    };
    match a {
        [x, y] => match (shape(x), shape(y)) {
            (Some((r1, c1)), Some((r2, c2))) => {
                if !r1.agrees(r2) {
                    return Err(format!("the two have {r1} and {r2} rows"));
                }
                Ok(Ty::Matrix(r1.meet(r2), add_sizes(c1, c2)))
            }
            _ => Err(bad_args("append_col(matrix|vector a, matrix|vector b)", a)),
        },
        _ => Err(bad_args("append_col(matrix|vector a, matrix|vector b)", a)),
    }
}

fn identity_sig(a: &[Ty], k: Consts, _: Option<usize>) -> Result<Ty, String> {
    match a {
        [Ty::Int] => {
            let n = const_size(k[0])?;
            Ok(Ty::Matrix(n, n))
        }
        _ => Err(bad_args("identity_matrix(int n)", a)),
    }
}

fn diag_matrix_sig(a: &[Ty], _: Consts, _: Option<usize>) -> Result<Ty, String> {
    match a {
        [Ty::Vector(n)] => Ok(Ty::Matrix(*n, *n)),
        _ => Err(bad_args("diag_matrix(vector v)", a)),
    }
}

fn diagonal_sig(a: &[Ty], _: Consts, _: Option<usize>) -> Result<Ty, String> {
    match a {
        [Ty::Matrix(r, c)] => Ok(Ty::Vector(match (r, c) {
            (Size::Known(r), Size::Known(c)) => Size::Known((*r).min(*c)),
            _ => Size::Unknown,
        })),
        _ => Err(bad_args("diagonal(matrix m)", a)),
    }
}

fn matrix_to_real_sig(a: &[Ty], _: Consts, _: Option<usize>) -> Result<Ty, String> {
    match a {
        [Ty::Matrix(..)] => Ok(Ty::Real),
        _ => Err(bad_args("f(matrix m)", a)),
    }
}

fn square_to_real_sig(a: &[Ty], _: Consts, _: Option<usize>) -> Result<Ty, String> {
    match a {
        [Ty::Matrix(r, c)] => square(*r, *c).map(|_| Ty::Real),
        _ => Err(bad_args("f(matrix m)", a)),
    }
}

fn square_sig(a: &[Ty], _: Consts, _: Option<usize>) -> Result<Ty, String> {
    match a {
        [Ty::Matrix(r, c)] => square(*r, *c).map(|n| Ty::Matrix(n, n)),
        _ => Err(bad_args("f(matrix m)", a)),
    }
}

fn eigenvalues_sig(a: &[Ty], _: Consts, _: Option<usize>) -> Result<Ty, String> {
    match a {
        [Ty::Matrix(r, c)] => square(*r, *c).map(Ty::Vector),
        _ => Err(bad_args("eigenvalues_sym(matrix m)", a)),
    }
}

fn crossprod_sig(a: &[Ty], _: Consts, _: Option<usize>) -> Result<Ty, String> {
    match a {
        [Ty::Matrix(_, c)] => Ok(Ty::Matrix(*c, *c)),
        _ => Err(bad_args("crossprod(matrix m)", a)),
    }
}

fn tcrossprod_sig(a: &[Ty], _: Consts, _: Option<usize>) -> Result<Ty, String> {
    match a {
        [Ty::Matrix(r, _)] => Ok(Ty::Matrix(*r, *r)),
        _ => Err(bad_args("f(matrix m)", a)),
    }
}

fn quad_form_sig(a: &[Ty], _: Consts, _: Option<usize>) -> Result<Ty, String> {
    match a {
        [Ty::Matrix(r, c), Ty::Vector(n)] => {
            let s = square(*r, *c)?;
            if !s.agrees(*n) {
                return Err(format!("a {s} x {s} matrix with a vector of {n}"));
            }
            Ok(Ty::Real)
        }
        _ => Err(bad_args("quad_form(matrix A, vector v)", a)),
    }
}

fn mdivide_sig(a: &[Ty], _: Consts, _: Option<usize>) -> Result<Ty, String> {
    let (r, c, b) = match a {
        [Ty::Matrix(r, c), b @ (Ty::Vector(_) | Ty::Matrix(..))] => (*r, *c, b),
        _ => {
            return Err(bad_args(
                "f(matrix A, vector b) or f(matrix A, matrix B)",
                a,
            ))
        }
    };
    let s = square(r, c)?;
    let rows = match b {
        Ty::Vector(n) => *n,
        Ty::Matrix(n, _) => *n,
        _ => unreachable!(),
    };
    if !s.agrees(rows) {
        return Err(format!(
            "a {s} x {s} matrix with a right-hand side of {rows} rows"
        ));
    }
    Ok(b.clone())
}

fn rng1_sig(a: &[Ty], _: Consts, _: Option<usize>) -> Result<Ty, String> {
    broadcast_n(a, 1, "f(real|vector x)")
}

fn rng2_sig(a: &[Ty], _: Consts, _: Option<usize>) -> Result<Ty, String> {
    broadcast_n(a, 2, "f(real|vector a, real|vector b)")
}

fn rng3_sig(a: &[Ty], _: Consts, _: Option<usize>) -> Result<Ty, String> {
    broadcast_n(a, 3, "f(real|vector a, real|vector b, real|vector c)")
}

fn bern_rng_sig(a: &[Ty], _: Consts, _: Option<usize>) -> Result<Ty, String> {
    match a {
        [t] if t.is_scalar() => Ok(Ty::Int),
        _ => Err(bad_args("f(real x)", a)),
    }
}

fn binomial_rng_sig(a: &[Ty], _: Consts, _: Option<usize>) -> Result<Ty, String> {
    match a {
        [Ty::Int, t] if t.is_scalar() => Ok(Ty::Int),
        _ => Err(bad_args("binomial_rng(int N, real theta)", a)),
    }
}

fn categorical_rng_sig(a: &[Ty], _: Consts, _: Option<usize>) -> Result<Ty, String> {
    match a {
        [Ty::Vector(_)] => Ok(Ty::Int),
        _ => Err(bad_args("f(vector theta)", a)),
    }
}

fn discrete_range_rng_sig(a: &[Ty], k: Consts, _: Option<usize>) -> Result<Ty, String> {
    match a {
        [Ty::Int, Ty::Int] => {
            if let (Some(lo), Some(hi)) = (k[0], k[1]) {
                if lo > hi {
                    return Err(format!("lower {lo} is above upper {hi}"));
                }
            }
            Ok(Ty::Int)
        }
        _ => Err(bad_args("discrete_range_rng(int lower, int upper)", a)),
    }
}

fn multi_normal_sig(a: &[Ty], _: Consts, _: Option<usize>) -> Result<Ty, String> {
    match a {
        [Ty::Vector(n), Ty::Matrix(r, c)] => {
            let s = square(*r, *c)?;
            if !s.agrees(*n) {
                return Err(format!("mu has {n} elements, the matrix is {s} x {s}"));
            }
            Ok(Ty::Vector(n.meet(s)))
        }
        _ => Err(bad_args("f(vector mu, matrix Sigma)", a)),
    }
}

fn multi_normal_lpdf_sig(a: &[Ty], _: Consts, _: Option<usize>) -> Result<Ty, String> {
    match a {
        [Ty::Vector(y), mu @ Ty::Vector(_), m @ Ty::Matrix(..)] => {
            let Ty::Vector(n) = multi_normal_sig(&[mu.clone(), m.clone()], &[None, None], None)?
            else {
                unreachable!()
            };
            if !y.agrees(n) {
                return Err(format!("y has {y} elements, mu {n}"));
            }
            Ok(Ty::Real)
        }
        _ => Err(bad_args("f(vector y | vector mu, matrix Sigma)", a)),
    }
}

fn dens1_sig(a: &[Ty], _: Consts, _: Option<usize>) -> Result<Ty, String> {
    broadcast_n(a, 1, "f(real|vector y)").map(|_| Ty::Real)
}

fn dens2_sig(a: &[Ty], _: Consts, _: Option<usize>) -> Result<Ty, String> {
    broadcast_n(a, 2, "f(real|vector y | real|vector a)").map(|_| Ty::Real)
}

fn dens3_sig(a: &[Ty], _: Consts, _: Option<usize>) -> Result<Ty, String> {
    broadcast_n(a, 3, "f(real|vector y | real|vector a, real|vector b)").map(|_| Ty::Real)
}

fn dens4_sig(a: &[Ty], _: Consts, _: Option<usize>) -> Result<Ty, String> {
    broadcast_n(
        a,
        4,
        "f(real|vector y | real|vector a, real|vector b, real|vector c)",
    )
    .map(|_| Ty::Real)
}

fn int_pmf_sig(a: &[Ty], _: Consts, _: Option<usize>) -> Result<Ty, String> {
    match a {
        [Ty::Int, t] if t.is_scalar() => Ok(Ty::Real),
        _ => Err(bad_args("f(int y | real theta)", a)),
    }
}

fn binomial_lpmf_sig(a: &[Ty], _: Consts, _: Option<usize>) -> Result<Ty, String> {
    match a {
        [Ty::Int, Ty::Int, t] if t.is_scalar() => Ok(Ty::Real),
        _ => Err(bad_args("binomial_lpmf(int n | int N, real theta)", a)),
    }
}

fn categorical_lpmf_sig(a: &[Ty], _: Consts, _: Option<usize>) -> Result<Ty, String> {
    match a {
        [Ty::Int, Ty::Vector(_)] => Ok(Ty::Real),
        _ => Err(bad_args("f(int k | vector theta)", a)),
    }
}
