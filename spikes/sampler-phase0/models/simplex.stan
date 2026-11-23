data {
  int<lower=2> K;
  vector<lower=0>[K] alpha;
}
parameters {
  simplex[K] theta;
}
model {
  theta ~ dirichlet(alpha);
}
