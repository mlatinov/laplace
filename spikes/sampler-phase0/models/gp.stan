functions {
// @laplace
// @brief Squared-exponential (RBF) covariance matrix
// @param x Vector of 1D input locations
// @param alpha Marginal standard deviation
// @param rho Length-scale
// @return NxN covariance matrix
// @example gps::rbf_cov(x, 1.0, 2.0)
matrix gps__rbf_cov(vector x, real alpha, real rho) {
  int N = num_elements(x);
  matrix[N, N] K;
  for (i in 1:N) {
    for (j in 1:N) {
      K[i, j] = square(alpha) * exp(-square(x[i] - x[j]) / (2 * square(rho)));
    }
  }
  return K;
}


}


data {
  int<lower=1> N;
  vector[N] x;
  vector[N] y;
}

parameters {
  real<lower=0> alpha;
  real<lower=0> rho;
  real<lower=0> sigma;
}

model {
  matrix[N, N] K = gps__rbf_cov(x, alpha, rho) + diag_matrix(rep_vector(square(sigma), N));
  alpha ~ normal(0, 1);
  rho ~ normal(0, 1);
  sigma ~ normal(0, 1);
  y ~ multi_normal(rep_vector(0, N), K);
}
