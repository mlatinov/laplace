// Phase 1 reference target: 10-dimensional correlated Gaussian, condition
// number 100. Sigma = Q diag(lambda) Q^T with lambda log-spaced on [1, 100]
// and Q a random rotation (make_corr_gauss_10.R), so a diagonal metric cannot
// undo the correlation. Truth: E[x_i] = 0, E[x_i^2] = Sigma[i, i].
data {
  int<lower=1> N;
  matrix[N, N] Sigma;
}
transformed data {
  matrix[N, N] L = cholesky_decompose(Sigma);
}
parameters {
  vector[N] x;
}
model {
  x ~ multi_normal_cholesky(rep_vector(0, N), L);
}
