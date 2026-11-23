// Phase 3 target: corr_gauss_10 (rotated, condition number 100) with its
// coordinates split across two parameters, so two blocks are strongly
// correlated with each other. Same data file as corr_gauss_10.
// Truth: E[x_i] = 0, E[x_i^2] = Sigma[i, i] for x = append_row(a, b).
data {
  int<lower=1> N;
  matrix[N, N] Sigma;
}
transformed data {
  matrix[N, N] L = cholesky_decompose(Sigma);
}
parameters {
  vector[5] a;
  vector[N - 5] b;
}
model {
  append_row(a, b) ~ multi_normal_cholesky(rep_vector(0, N), L);
}
