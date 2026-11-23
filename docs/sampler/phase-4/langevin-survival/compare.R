# Score each run in runs/<name>/ against reference.json (|z| < 4 with both
# MCSEs, R-hat < 1.01, bulk ESS > 400), and report ESS per gradient and per
# second of sampling (summed over chains, from laplace sample's stderr).
suppressPackageStartupMessages({ library(cmdstanr); library(posterior); library(jsonlite) })
ref <- fromJSON("reference.json")$quantities
rows <- list()
runs <- commandArgs(TRUE); if (!length(runs)) runs <- c("nuts", "hybrid", "nuts_barker", "barker")
for (run in runs) {
  fit <- suppressWarnings(as_cmdstan_fit(list.files(file.path("runs", run), full.names = TRUE, pattern = "csv$")))
  d <- fit$draws(ref$name)
  z <- c(); rh <- c(); es <- c()
  for (i in seq_len(nrow(ref))) {
    x <- extract_variable_matrix(d, ref$name[i])
    z <- c(z, (mean(x) - ref$mean[i]) / sqrt(mcse_mean(x)^2 + ref$mcse_mean[i]^2),
              (mean(x^2) - ref$m2[i]) / sqrt(mcse_mean(x^2)^2 + ref$mcse_m2[i]^2))
    rh <- c(rh, rhat(x), rhat(x^2)); es <- c(es, ess_bulk(x), ess_bulk(x^2))
  }
  s <- fit$sampler_diagnostics(format = "df")
  grads <- sum(s$n_grad__); logps <- sum(s$n_logp__)
  err <- readLines(paste0("runs-", run, ".err"))
  secs <- sum(as.numeric(sub(".* ([0-9.]+) s sampling.*", "\\1", grep("s sampling", err, value = TRUE))))
  verdict <- if (max(rh) >= 1.01 || min(es) <= 400) "did-not-converge" else if (max(abs(z)) < 4) "pass" else "FAIL"
  rows[[run]] <- data.frame(run = run, max_abs_z = round(max(abs(z)), 2), max_rhat = round(max(rh), 4),
    min_ess = round(min(es)), grads = grads, logps = logps, secs = round(secs, 2),
    ess_per_grad = signif(min(es) / grads, 3), ess_per_s = round(min(es) / secs), verdict = verdict)
}
out <- do.call(rbind, rows)
out$grad_ratio <- round(out$ess_per_grad / out$ess_per_grad[1], 2)
out$s_ratio <- round(out$ess_per_s / out$ess_per_s[1], 2)
print(out, row.names = FALSE)
