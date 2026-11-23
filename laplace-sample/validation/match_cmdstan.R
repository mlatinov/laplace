# Phase 1.8: laplace-sample (nuts, mala) against a long CmdStan run on
# library models. Run from laplace-sample/validation with laplace-sample on PATH:
#   Rscript match_cmdstan.R [model ...]        (default: gp survival ts)
# Writes results/<model>.md and prints the same tables.
suppressPackageStartupMessages({ library(cmdstanr); library(posterior) })
options(width = 140)
models <- commandArgs(trailingOnly = TRUE)
if (!length(models)) models <- c("gp", "survival", "ts")
dir.create("results", showWarnings = FALSE)
dir.create("work", showWarnings = FALSE)
SEED_REF <- 1; SEED_STD <- 2; SEED_LS <- 20261123

elapsed_sampling <- function(f) {  # CmdStan "Elapsed Time: ... (Sampling)"
  l <- grep("(Sampling)", readLines(f), value = TRUE, fixed = TRUE)
  as.numeric(sub(".*?([0-9.]+) seconds \\(Sampling\\).*", "\\1", l))
}

for (m in models) {
  stan <- file.path("models", paste0(m, ".stan")); data <- file.path("models", paste0(m, ".json"))
  w <- file.path("work", m); unlink(w, recursive = TRUE); dir.create(w, recursive = TRUE)
  mod <- cmdstan_model(stan, dir = w, quiet = TRUE)
  v <- mod$variables()
  keep <- c("lp__", names(v$parameters), names(v$transformed_parameters))
  base <- function(x) sub("\\[.*", "", x)

  ref <- mod$sample(data = data, chains = 4, parallel_chains = 4, iter_warmup = 1000,
                    iter_sampling = 25000, seed = SEED_REF, refresh = 0, show_messages = FALSE)
  std <- mod$sample(data = data, chains = 4, parallel_chains = 4, iter_warmup = 1000,
                    iter_sampling = 1000, seed = SEED_STD, refresh = 0, show_messages = FALSE)
  rd <- ref$draws(); vars <- variables(rd)[base(variables(rd)) %in% keep]
  stats <- function(d) summarise_draws(subset_draws(d, variable = vars),
                                       mean, sd, mcse_mean, mcse_sd, rhat, ess_bulk)
  rs <- stats(rd)
  sd_std <- std$sampler_diagnostics()
  std_grads <- sum(extract_variable_matrix(sd_std, "n_leapfrog__"))
  std_ess <- min(stats(std$draws())$ess_bulk)
  std_time <- sum(sapply(std$output_files(), elapsed_sampling))

  out <- c(sprintf("# %s: laplace-sample vs CmdStan %s", m, cmdstan_version()), "",
           sprintf("Reference: CmdStan, 4 x 25000 draws (seed %d). Runs: 4 x 1000 warmup + 1000 draws.", SEED_REF), "")
  cost <- data.frame(sampler = "CmdStan NUTS", min_ess_bulk = std_ess, grads = std_grads,
                     ess_per_grad = std_ess / std_grads, us_per_grad = 1e6 * std_time / std_grads)
  for (s in c("nuts", "mala")) {
    od <- file.path(w, s); dir.create(od)
    err <- file.path(od, "stderr.txt")
    st <- system2("laplace-sample", c("sample", stan, "--data", data, "--sampler", s,
                                      "--seed", SEED_LS, "--output-dir", od), stdout = TRUE, stderr = err)
    if (!is.null(attr(st, "status"))) stop(paste(readLines(err), collapse = "\n"))
    fit <- as_cmdstan_fit(sort(list.files(od, "\\.csv$", full.names = TRUE)))
    ls <- stats(fit$draws())
    z_mean <- (ls$mean - rs$mean) / sqrt(ls$mcse_mean^2 + rs$mcse_mean^2)
    z_sd <- (ls$sd - rs$sd) / sqrt(ls$mcse_sd^2 + rs$mcse_sd^2)
    tab <- data.frame(variable = ls$variable, ref_mean = rs$mean, mean = ls$mean, z_mean = z_mean,
                      ref_sd = rs$sd, sd = ls$sd, z_sd = z_sd, rhat = ls$rhat, ess_bulk = ls$ess_bulk)
    grads <- sum(extract_variable_matrix(fit$sampler_diagnostics(), "n_grad__"))
    secs <- sum(as.numeric(sub(".* ([0-9.]+) s sampling.*", "\\1", grep("s sampling", readLines(err), value = TRUE))))
    cost <- rbind(cost, data.frame(sampler = paste("laplace-sample", s), min_ess_bulk = min(ls$ess_bulk),
                                   grads = grads, ess_per_grad = min(ls$ess_bulk) / grads,
                                   us_per_grad = 1e6 * secs / grads))
    maxz <- max(abs(c(z_mean, z_sd)))
    out <- c(out, sprintf("## %s — max |z| = %.2f (%s), max R-hat %.4f, min ESS %.0f", s, maxz,
                          if (maxz < 4) "PASS" else "FAIL", max(ls$rhat), min(ls$ess_bulk)), "",
             knitr::kable(tab, digits = 4), "")
  }
  out <- c(out, "## Cost (sampling phase only)", "",
           "`grads`: CmdStan n_leapfrog__ (one gradient each), laplace-sample n_grad__. `us_per_grad`: sampling wall time / grads, per chain summed, sampler included.", "",
           knitr::kable(cost, digits = c(0, 0, 0, 5, 1)), "")
  writeLines(out, file.path("results", paste0(m, ".md")))
  cat(out, sep = "\n")
}
