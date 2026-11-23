# 0.4 follow-up probes. Run after R/csv_header.R.
suppressPackageStartupMessages({ library(cmdstanr); library(posterior) })
d0 <- lapply(1:4, function(i) read.csv(sprintf("out/nutsrs/std_normal-%d.csv", i), check.names = FALSE))

# A clean, hand-written header (no indentation, no "(Default)").
header <- function(i, extra = character()) c(
  "# model = std_normal_model",
  "# method = sample",
  "# num_samples = 1000",
  "# save_warmup = false",
  "# thin = 1",
  "# metric = diag_e",
  sprintf("# id = %d", i),
  extra)
write_chains <- function(tag, cols_fn, extra = character()) {
  d <- file.path(tempdir(), tag); unlink(d, recursive = TRUE); dir.create(d)
  vapply(1:4, function(i) {
    x <- d0[[i]]; n <- nrow(x)
    diag <- cols_fn(x, n)
    out <- cbind(diag, x[, grep("^x\\.", names(x))])
    f <- file.path(d, sprintf("m-%d.csv", i))
    writeLines(c(header(i, extra), paste(names(out), collapse = ","),
                 apply(out, 1, function(r) paste(format(r, digits = 10, trim = TRUE), collapse = ","))), f)
    f
  }, "")
}
step <- function(label, expr) {
  r <- tryCatch({ force(expr); "ok" }, error = function(e) paste("ERROR:", conditionMessage(e)))
  cat(sprintf("   %-22s %s\n", label, r))
  invisible(r == "ok")
}
check <- function(title, files) {
  cat(title, "\n")
  fit <- NULL
  if (!step("as_cmdstan_fit", fit <- suppressWarnings(as_cmdstan_fit(files)))) return(invisible())
  step("$summary()", fit$summary())
  step("$draws()", fit$draws())
  step("$diagnostic_summary()", print(unlist(suppressWarnings(fit$diagnostic_summary(quiet = TRUE)))))
  step("$sampler_diagnostics()", fit$sampler_diagnostics())
  invisible(fit)
}
mala <- function(x, n) data.frame(lp__ = x$lp__, accept_stat__ = 0.57, stepsize__ = x$stepsize__,
                                  n_grad__ = x$n_grad__, check.names = FALSE)
nuts_fill <- function(x, n, energy) data.frame(lp__ = x$lp__, accept_stat__ = 0.57, stepsize__ = x$stepsize__,
                                  treedepth__ = 0, n_leapfrog__ = x$n_grad__, divergent__ = 0,
                                  energy__ = energy, n_grad__ = x$n_grad__, check.names = FALSE)

check("Q1 hand-written 7-line header, NUTS-shaped columns",
      write_chains("q1", function(x, n) nuts_fill(x, n, -x$lp__ + rchisq(n, 10) / 2)))
check("Q2 non-NUTS columns (lp, accept, stepsize, n_grad): which call fails?",
      write_chains("q2", mala))
check("Q3 non-NUTS sampler padded: treedepth__=0, n_leapfrog__=n_grad, divergent__=0, energy__=NaN",
      write_chains("q3", function(x, n) nuts_fill(x, n, NaN)))
check("Q4 as Q3 but energy__ = -lp__ (not a real Hamiltonian; E-BFMI meaningless)",
      write_chains("q4", function(x, n) nuts_fill(x, n, -x$lp__)))
check("Q5 Q1 + '# algorithm = laplace'",
      write_chains("q5", function(x, n) nuts_fill(x, n, -x$lp__ + rchisq(n, 10) / 2), "# algorithm = laplace"))
check("Q6 Q2 + '# algorithm = fixed_param' (the other built-in path)",
      write_chains("q6", mala, "# algorithm = fixed_param"))
f <- write_chains("q7", function(x, n) nuts_fill(x, n, -x$lp__ + rchisq(n, 10) / 2),
                  c("# laplace_sampler = hybrid", "# laplace_kernel_beta = nuts() builtin",
                    "# laplace_kernel_sigma_tau = langevin::mala 0.1.0 sha256:89ab"))
fit <- check("Q7 one key per kernel: laplace_kernel_<label>", f)
md <- fit$metadata(); cat("   metadata keys:", paste(grep("^laplace", names(md), value = TRUE), collapse = ", "), "\n")
