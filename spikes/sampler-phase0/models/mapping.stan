// Phase 0.2: one of each constrained shape, plus TP and GQ with an RNG.
parameters {
  real mu;
  real<lower=0> sigma;
  vector[3] beta;
  simplex[4] theta;
  cholesky_factor_corr[3] L;
  matrix[2, 2] M;
  array[2] real<lower=-1, upper=1> r;
}
transformed parameters {
  real log_sigma = log(sigma);
  vector[3] beta_scaled = sigma * beta;
}
model {
  mu ~ normal(0, 1);
  sigma ~ lognormal(0, 0.5);
  beta ~ std_normal();
  theta ~ dirichlet(rep_vector(2, 4));
  L ~ lkj_corr_cholesky(2);
  to_vector(M) ~ std_normal();
}
generated quantities {
  real y_rep = normal_rng(mu, sigma);
  int k = categorical_rng(theta);
}
