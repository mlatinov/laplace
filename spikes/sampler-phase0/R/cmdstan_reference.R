# Phase 0 CmdStan references: std_normal (0.4 CSV, 0.5 moments) and gp (0.6 timing).
# Run from spikes/sampler-phase0:  Rscript R/cmdstan_reference.R
suppressPackageStartupMessages(library(cmdstanr))
dir.create("out/cmdstan", recursive = TRUE, showWarnings = FALSE)
dir.create("out/exe", recursive = TRUE, showWarnings = FALSE)

# GP data: fixed seed, written once and shared with BridgeStan.
set.seed(1)
N <- 40
x <- seq(-5, 5, length.out = N)
y <- sin(x) + rnorm(N, 0, 0.3)
write_stan_json(list(N = N, x = x, y = y), "models/gp.json")

run <- function(name, data = NULL, iter = 1000, seed = 1, basename = name) {
  m <- cmdstan_model(file.path("models", paste0(name, ".stan")), dir = "out/exe", quiet = TRUE)
  fit <- m$sample(data = data, chains = 4, parallel_chains = 4,
                  iter_warmup = iter, iter_sampling = iter, seed = seed,
                  output_dir = "out/cmdstan", output_basename = basename,
                  refresh = 0, show_messages = FALSE)
  fit
}

sn <- run("std_normal")
sn_long <- run("std_normal", iter = 10000, seed = 2, basename = "std_normal_long")  # long run for 0.5 comparison
gp <- run("gp", data = "models/gp.json")

print(sn$summary())
print(gp$summary())
print(gp$diagnostic_summary())
cat("\nstd_normal csvs:", sn$output_files(), sep = "\n")
cat("gp csvs:", gp$output_files(), sep = "\n")
