// A bimodal Gaussian mixture, 2 dimensions (reference suite, report only):
// half N(-2, I), half N(+2, I). Truth: the mass with x[1] + x[2] > 0 is
// exactly 0.5 (by symmetry).
parameters {
  vector[2] x;
}
model {
  target += log_mix(0.5, normal_lpdf(x | -2, 1), normal_lpdf(x | 2, 1));
}
