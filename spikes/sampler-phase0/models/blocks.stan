// 0.5 block-sweep target. (a[i], b[i]) bivariate normal: sd(a)=1, sd(b)=s[i],
// corr = rho. Conditional a | b has a fixed covariance and a moving mean, so a
// correctly adapted block sampler for `a` should settle on one step size.
transformed data {
  vector[3] s = [1, 5, 0.2]';
  real rho = 0.9;
}
parameters {
  vector[3] a;
  vector[3] b;
}
model {
  a ~ std_normal();
  b ~ normal(s .* (rho * a), s * sqrt(1 - rho^2));
}
