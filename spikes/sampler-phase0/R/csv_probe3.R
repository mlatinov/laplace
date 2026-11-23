# 0.4 follow-up after review (Q4): non-NUTS runs write NaN in every NUTS column
# that does not apply. Does $summary(), $draws(), $diagnostic_summary() still run?
# Run from spikes/sampler-phase0:  Rscript R/csv_probe3.R
suppressPackageStartupMessages({ library(cmdstanr); library(posterior) })
d0 <- lapply(1:4, function(i) read.csv(sprintf("out/nutsrs/std_normal-%d.csv", i), check.names = FALSE))
na_cols <- c("treedepth__", "n_leapfrog__", "divergent__", "energy__")

header <- function(i) c("# model = std_normal_model", "# method = sample", "# num_samples = 1000",
                        "# save_warmup = false", "# thin = 1", "# metric = diag_e", sprintf("# id = %d", i),
                        "# laplace_sampler = mala (sha256: 0123abcd)",
                        "# laplace_kernel_x = mala builtin 0.1.0")
write_chains <- function(tag, nan_cols) {
  d <- file.path(tempdir(), tag); unlink(d, recursive = TRUE); dir.create(d)
  vapply(1:4, function(i) {
    x <- d0[[i]]
    diag <- data.frame(lp__ = x$lp__, accept_stat__ = 0.57, stepsize__ = x$stepsize__,
                       treedepth__ = 0, n_leapfrog__ = x$n_grad__, divergent__ = 0,
                       energy__ = -x$lp__, n_grad__ = x$n_grad__, check.names = FALSE)
    for (c in nan_cols) diag[[c]] <- NaN
    out <- cbind(diag, x[, grep("^x\\.", names(x))])
    f <- file.path(d, sprintf("m-%d.csv", i))
    writeLines(c(header(i), paste(names(out), collapse = ","),
                 apply(out, 1, function(r) paste(format(r, digits = 10, trim = TRUE), collapse = ","))), f)
    f
  }, "")
}
run <- function(label, nan_cols) {
  files <- write_chains(gsub("[^a-z]", "", tolower(label)), nan_cols)
  res <- list()
  warns <- character()
  step <- function(name, expr) {
    res[[name]] <<- withCallingHandlers(
      tryCatch({ v <- force(expr); "ok" }, error = function(e) paste("ERROR:", conditionMessage(e))),
      warning = function(w) { warns <<- c(warns, conditionMessage(w)); invokeRestart("muffleWarning") },
      message = function(m) invokeRestart("muffleMessage"))
  }
  fit <- NULL
  step("as_cmdstan_fit", fit <- as_cmdstan_fit(files))
  ds <- NULL
  if (!is.null(fit)) {
    step("$summary()", { s <- fit$summary(); stopifnot(!anyNA(s$mean)); s })
    step("$draws()", fit$draws())
    step("$diagnostic_summary()", ds <- fit$diagnostic_summary(quiet = TRUE))
    step("$sampler_diagnostics()", fit$sampler_diagnostics())
  }
  cat(sprintf("\n== %s  (NaN in: %s)\n", label, if (length(nan_cols)) paste(nan_cols, collapse = ", ") else "none"))
  for (n in names(res)) cat(sprintf("   %-24s %s\n", n, res[[n]]))
  if (!is.null(ds)) cat("   diagnostic_summary:", paste(names(unlist(ds)), unlist(ds), sep = "=", collapse = " "), "\n")
  if (length(warns)) cat("   warnings:", paste(unique(warns), collapse = " | "), "\n")
  invisible(all(unlist(res) == "ok"))
}
all_ok <- run("R1 all four NUTS columns NaN", na_cols)
for (c in na_cols) run(paste("R2 only", c, "NaN"), c)
cat(sprintf("\nRESULT: NaN in all of {%s} %s\n", paste(na_cols, collapse = ", "),
            if (all_ok) "works for every call" else "breaks something; see per-column runs"))
