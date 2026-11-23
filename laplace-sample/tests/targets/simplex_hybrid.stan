// Phase 3 target: a simplex in one block, a vector depending on it in the
// other. theta ~ Dirichlet(alpha) and beta | theta ~ N(2 theta[1:3], 1):
//   E[theta_i]    = alpha_i / A                     (A = sum(alpha) = 14)
//   E[theta_i^2]  = alpha_i (alpha_i + 1) / (A (A + 1))
//   E[beta_i]     = 2 E[theta_i]
//   E[beta_i^2]   = 1 + 4 E[theta_i^2]
// The simplex[4] has 3 unconstrained coordinates (Phase 0.2 mapping).
parameters {
  simplex[4] theta;
  vector[3] beta;
}
model {
  theta ~ dirichlet([2, 3, 4, 5]');
  beta ~ normal(2 * theta[1:3], 1);
}
