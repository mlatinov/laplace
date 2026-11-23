// A banana (Rosenbrock-type) target, 2 dimensions (reference suite, report
// only). x1 ~ N(0, 1), x2 | x1 ~ N(x1^2, 0.5).
// Truth: E[x1] = 0, E[x1^2] = 1, E[x2] = 1, E[x2^2] = 0.25 + 3 = 3.25.
parameters {
  real x1;
  real x2;
}
model {
  x1 ~ normal(0, 1);
  x2 ~ normal(square(x1), 0.5);
}
