# Phase 1.7 check: cmdstanr::as_cmdstan_fit() reads laplace-sample output.
# Usage: Rscript check_csv.R <runs dir>   (one subdirectory per sampler)
suppressPackageStartupMessages({ library(cmdstanr); library(posterior) })
args <- commandArgs(trailingOnly = TRUE)
runs <- if (length(args)) args[1] else "runs"
ok_all <- TRUE
for (s in sort(list.dirs(runs, recursive = FALSE, full.names = FALSE))) {
  files <- sort(list.files(file.path(runs, s), "\\.csv$", full.names = TRUE))
  res <- tryCatch({
    fit <- suppressWarnings(as_cmdstan_fit(files))
    sm <- fit$summary()
    dr <- fit$draws()
    ds <- suppressWarnings(fit$diagnostic_summary(quiet = TRUE))
    md <- fit$metadata()
    stopifnot(!anyNA(sm$mean), dim(dr)[2] == length(files))
    cat(sprintf("== %s: as_cmdstan_fit OK, %d chains x %d draws, %d variables\n",
                s, dim(dr)[2], dim(dr)[1], nrow(sm)))
    cat("   diagnostic_summary:", paste(names(unlist(ds)), signif(unlist(ds), 3), sep = "=", collapse = " "), "\n")
    cat("   metadata: laplace_sampler =", md$laplace_sampler, "\n")
    cat("             laplace_kernel_rest =", md$laplace_kernel_rest, "\n")
    cat("             step_size_adaptation =", md$step_size_adaptation, "\n")
    im <- tryCatch(fit$inv_metric(), error = function(e) NULL)
    if (!is.null(im)) cat("   inv_metric(chain 1):", signif(diag(as.matrix(im[[1]])), 3), "\n") else
      cat("   inv_metric: none (several blocks, each with its own; see stepsize_<label>__)\n")
    cat("   sampler_diagnostics:", paste(dimnames(fit$sampler_diagnostics())$variable, collapse = ","), "\n")
    print(sm[, c("variable", "mean", "sd", "rhat", "ess_bulk")], n = 20)
    TRUE
  }, error = function(e) { cat(sprintf("== %s: FAILED: %s\n", s, conditionMessage(e))); FALSE })
  ok_all <- ok_all && res
}
cat(if (ok_all) "\nALL OK\n" else "\nSOME FAILED\n")
quit(status = if (ok_all) 0 else 1)
