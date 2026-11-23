# Writes corr_gauss_10.json. Run from this directory: Rscript make_corr_gauss_10.R
set.seed(20261123)
N <- 10
Q <- qr.Q(qr(matrix(rnorm(N * N), N)))
lambda <- 10^seq(0, 2, length.out = N)
Sigma <- Q %*% diag(lambda) %*% t(Q)
Sigma <- (Sigma + t(Sigma)) / 2
stopifnot(abs(kappa(Sigma, exact = TRUE) - 100) < 1e-6)
jsonlite::write_json(list(N = N, Sigma = Sigma), "corr_gauss_10.json", digits = NA, auto_unbox = TRUE)
