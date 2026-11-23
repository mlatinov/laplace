# Long CmdStan reference for build/model.stan + data.json: E[x], E[x^2] and
# MCSEs for every parameter. Run after `laplace build model.laplace`.
suppressPackageStartupMessages({ library(cmdstanr); library(posterior) })
m <- cmdstan_model("build/model.stan", dir = tempdir(), quiet = TRUE)
fit <- m$sample(data = "data.json", chains = 4, parallel_chains = 4, iter_warmup = 2000,
                iter_sampling = 25000, seed = 1, refresh = 0, show_messages = FALSE)
d <- fit$draws(c("a", "gamma", "eta0", "beta"))
ref <- lapply(variables(d), function(v) { x <- extract_variable_matrix(d, v)
  list(name = v, mean = mean(x), mcse_mean = mcse_mean(x), m2 = mean(x^2), mcse_m2 = mcse_mean(x^2),
       rhat = rhat(x), ess_bulk = ess_bulk(x)) })
jsonlite::write_json(list(cmdstan = cmdstan_version(), draws = "4 x 25000", seed = 1,
  divergences = sum(fit$diagnostic_summary(quiet = TRUE)$num_divergent), quantities = ref),
  "reference.json", digits = NA, auto_unbox = TRUE, pretty = TRUE)
cat("divergences:", sum(fit$diagnostic_summary(quiet = TRUE)$num_divergent),
    " max rhat:", max(sapply(ref, `[[`, "rhat")), " min ess:", min(sapply(ref, `[[`, "ess_bulk")), "\n")
print(do.call(rbind, lapply(ref, function(r) data.frame(name = r$name, mean = signif(r$mean, 4), sd = signif(sqrt(r$m2 - r$mean^2), 3)))))
