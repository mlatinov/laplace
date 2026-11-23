# Writes eight_schools_nc.reference.json: a long CmdStan run of the
# non-centred eight schools model (4 x 25000 draws, adapt_delta 0.95,
# seed 1), with E[x] and E[x^2] and their MCSEs for every parameter and
# transformed parameter. Run from this directory:
#   Rscript make_eight_schools_reference.R
suppressPackageStartupMessages({ library(cmdstanr); library(posterior) })
m <- cmdstan_model("eight_schools_nc.stan", dir = tempdir(), quiet = TRUE)
fit <- m$sample(data = "eight_schools_nc.json", chains = 4, parallel_chains = 4,
                iter_warmup = 2000, iter_sampling = 25000, adapt_delta = 0.95,
                seed = 1, refresh = 0, show_messages = FALSE)
d <- fit$draws(c("mu", "tau", "theta_tilde", "theta"))
ref <- lapply(variables(d), function(v) {
  x <- extract_variable_matrix(d, v)
  list(name = v, mean = mean(x), mcse_mean = mcse_mean(x),
       m2 = mean(x^2), mcse_m2 = mcse_mean(x^2), rhat = rhat(x), ess_bulk = ess_bulk(x))
})
ds <- fit$diagnostic_summary(quiet = TRUE)
out <- list(cmdstan = cmdstan_version(), draws = "4 x 25000", adapt_delta = 0.95, seed = 1,
            divergences = sum(ds$num_divergent), quantities = ref)
jsonlite::write_json(out, "eight_schools_nc.reference.json", digits = NA, auto_unbox = TRUE, pretty = TRUE)
cat("divergences:", sum(ds$num_divergent), " max rhat:", max(sapply(ref, `[[`, "rhat")),
    " min ess_bulk:", min(sapply(ref, `[[`, "ess_bulk")), "\n")
