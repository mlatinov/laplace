// Neal's funnel, 10 dimensions (reference suite, report only).
// Truth: E[v] = 0, E[v^2] = 9 (the exact variance of the log scale).
parameters {
  real v;
  vector[9] x;
}
model {
  v ~ normal(0, 3);
  x ~ normal(0, exp(v / 2));
}
