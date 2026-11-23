# 0.5 moment checks. Run from spikes/sampler-phase0:  Rscript R/check_nuts.R
suppressPackageStartupMessages({ library(posterior); library(cmdstanr) })

read_plain <- function(pattern, dir) {
  files <- sort(list.files(dir, pattern = pattern, full.names = TRUE))
  dfs <- lapply(seq_along(files), function(i) {
    d <- fix_names(read.csv(files[i], check.names = FALSE)); d$.chain <- i; d$.iteration <- seq_len(nrow(d)); d
  })
  as_draws_df(do.call(rbind, dfs))
}
fix_names <- function(d) {  # x.1 -> x[1]
  names(d) <- sub("^([A-Za-z_][A-Za-z0-9_]*)\\.(.*)$", "\\1[\\2]", names(d))
  names(d) <- gsub("\\.", ",", names(d)); d
}
# z for E[x] and E[x^2]; MCSE from posterior (bulk-ESS based).
moments <- function(d, vars) {
  do.call(rbind, lapply(vars, function(v) {
    x <- extract_variable_matrix(d, v)
    data.frame(variable = v,
               mean = mean(x), mcse_mean = mcse_mean(x),
               m2 = mean(x^2), mcse_m2 = mcse_mean(x^2),
               rhat = rhat(x), ess_bulk = ess_bulk(x))
  }))
}
report <- function(title, tab) {
  cat("\n==", title, "\n"); print(tab, digits = 3, row.names = FALSE)
  z <- unlist(tab[grep("^z_", names(tab))])
  cat(sprintf("max |z| = %.2f -> %s; max rhat = %.4f; min ess_bulk = %.0f\n",
              max(abs(z)), if (max(abs(z)) < 4) "PASS" else "FAIL",
              max(tab$rhat), min(tab$ess_bulk)))
}

# --- A1: std_normal, nuts-rs vs exact moments and vs long CmdStan run
nr <- (read_plain("^std_normal-", "out/nutsrs"))
vars <- paste0("x[", 1:10, "]")
t1 <- moments(nr, vars)
t1$z_mean <- t1$mean / t1$mcse_mean
t1$z_m2 <- (t1$m2 - 1) / t1$mcse_m2
report("A1 std_normal nuts-rs vs exact", t1)
ngrad <- sum(nr$n_grad__)
cat(sprintf("ESS/grad (min bulk) = %.4f\n", min(t1$ess_bulk) / ngrad))

cs_long <- as_draws_df(read_cmdstan_csv(list.files("out/cmdstan", "^std_normal_long-", full.names = TRUE))$post_warmup_draws)
tl <- moments(cs_long, vars)
t1b <- data.frame(variable = vars, rhat = t1$rhat, ess_bulk = t1$ess_bulk,
                  z_mean = (t1$mean - tl$mean) / sqrt(t1$mcse_mean^2 + tl$mcse_mean^2),
                  z_m2 = (t1$m2 - tl$m2) / sqrt(t1$mcse_m2^2 + tl$mcse_m2^2))
report("A1' std_normal nuts-rs vs CmdStan 4x10000", t1b)

# --- A2: gp, nuts-rs vs CmdStan (same data)
ng <- (read_plain("^gp-", "out/nutsrs"))
cg <- as_draws_df(read_cmdstan_csv(list.files("out/cmdstan", "^gp-", full.names = TRUE))$post_warmup_draws)
gv <- c("alpha", "rho", "sigma")
a <- moments(ng, gv); b <- moments(cg, gv)
t2 <- data.frame(variable = gv, nutsrs_mean = a$mean, cmdstan_mean = b$mean,
                 rhat = a$rhat, ess_bulk = a$ess_bulk,
                 z_mean = (a$mean - b$mean) / sqrt(a$mcse_mean^2 + b$mcse_mean^2),
                 z_m2 = (a$m2 - b$m2) / sqrt(a$mcse_m2^2 + b$mcse_m2^2))
report("A2 gp nuts-rs vs CmdStan", t2)
cg_diag <- read_cmdstan_csv(list.files("out/cmdstan", "^gp-", full.names = TRUE))$post_warmup_sampler_diagnostics
cmd_grad <- sum(extract_variable(cg_diag, "n_leapfrog__"))
cat(sprintf("ESS/grad (min bulk): nuts-rs %.4f (%d grads), CmdStan %.4f (%d leapfrogs)\n",
            min(a$ess_bulk) / sum(ng$n_grad__), sum(ng$n_grad__),
            min(b$ess_bulk) / cmd_grad, cmd_grad))

# --- B: Gibbs variants vs exact moments
s <- c(1, 5, 0.2)
bv <- c(paste0("a[", 1:3, "]"), paste0("b[", 1:3, "]"))
truth_m2 <- c(1, 1, 1, s^2)
for (v in c("SetPosition", "Stale", "Refresh")) {
  d <- (read_plain(paste0("^", v, "-"), "out/gibbs"))
  t <- moments(d, bv)
  t$z_mean <- t$mean / t$mcse_mean
  t$z_m2 <- (t$m2 - truth_m2) / t$mcse_m2
  report(paste("B Gibbs", v), t)
  cat(sprintf("total grads (sampling) = %d; ESS/grad (min bulk) = %.5f\n",
              sum(d$n_grad__), min(t$ess_bulk) / sum(d$n_grad__)))
}
