# 0.4 CSV compatibility. Run from spikes/sampler-phase0:  Rscript R/csv_header.R
# Finds the minimal comment header for which cmdstanr::as_cmdstan_fit() works,
# then probes laplace's own additions. Writes the working example to
# out/csv/example-*.csv.
suppressPackageStartupMessages({ library(cmdstanr); library(posterior) })
options(width = 120)
ref_files <- sort(list.files("out/cmdstan", "^std_normal-[0-9]\\.csv$", full.names = TRUE))
ref <- lapply(ref_files, readLines)
ref_fit <- as_cmdstan_fit(ref_files)
ref_sum <- ref_fit$summary()

# Does a set of 4 chain files (character vectors) pass?  Returns "" or a reason.
probe <- function(chains, tag) {
  d <- file.path(tempdir(), tag); unlink(d, recursive = TRUE); dir.create(d)
  files <- file.path(d, sprintf("m-%d.csv", seq_along(chains)))
  for (i in seq_along(chains)) writeLines(chains[[i]], files[i])
  warns <- character()
  res <- withCallingHandlers(tryCatch({
    fit <- as_cmdstan_fit(files)
    s <- fit$summary()
    dr <- fit$draws()
    ds <- fit$diagnostic_summary(quiet = TRUE)
    if (!all(c("num_divergent") %in% names(ds))) stop("diagnostic_summary lacks num_divergent")
    if (nrow(s) < 2 || any(is.na(s$mean))) stop("summary broken")
    if (dim(dr)[2] != length(chains)) stop("draws chain dim wrong")
    list(ok = TRUE, fit = fit, s = s, ds = ds)
  }, error = function(e) list(ok = FALSE, why = conditionMessage(e))),
  warning = function(w) { warns <<- c(warns, conditionMessage(w)); invokeRestart("muffleWarning") },
  message = function(m) invokeRestart("muffleMessage"))
  res$warns <- warns
  res
}
is_comment <- function(x) startsWith(x, "#")

# ---- 1. Greedy minimisation of the comment lines (applied to all 4 chains alike).
# The inverse-metric label and its value row are removed as one unit.
hdr_idx <- which(is_comment(ref[[1]]))
units <- as.list(hdr_idx)
lab <- grep("Diagonal elements of inverse mass matrix", ref[[1]])
units <- Filter(function(u) !(u %in% c(lab, lab + 1)), units)
units <- c(units, list(c(lab, lab + 1)))
base_warns <- probe(ref, "base")$warns
drop <- integer()
for (u in units) {
  try_drop <- c(drop, u)
  ch <- lapply(ref, function(l) l[-try_drop])
  r <- probe(ch, "greedy")
  if (isTRUE(r$ok) && length(setdiff(r$warns, base_warns)) == 0) drop <- try_drop
}
minimal <- lapply(ref, function(l) l[-drop])
cat("== Minimal header (chain 1):\n"); cat(minimal[[1]][is_comment(minimal[[1]])], sep = "\n")
rm <- probe(minimal, "minimal")
cat("minimal passes:", rm$ok, " summary identical to full:",
    isTRUE(all.equal(as.data.frame(rm$s), as.data.frame(ref_sum))), "\n")
cat("diagnostic_summary on minimal:\n"); str(rm$ds)

# ---- 2. Probes on top of the minimal header, using nuts-rs spike draws.
mk <- function(i, header, cols, extra_comments = character()) {
  d <- read.csv(sprintf("out/nutsrs/std_normal-%d.csv", i), check.names = FALSE)
  n <- nrow(d)
  set.seed(i)
  all <- data.frame(lp__ = d$lp__,
                    accept_stat__ = runif(n, 0.6, 1),  # nuts-rs Progress lacks it (spike filler)
                    stepsize__ = d$stepsize__,
                    treedepth__ = ceiling(log2(d$n_leapfrog__ + 1)),
                    n_leapfrog__ = d$n_leapfrog__,
                    divergent__ = d$divergent__,
                    energy__ = -d$lp__ + rchisq(n, 10) / 2,  # spike filler
                    n_grad__ = d$n_grad__,
                    accept_stat_beta__ = runif(n, 0.4, 0.8),
                    check.names = FALSE)
  all <- all[, cols, drop = FALSE]
  params <- d[, grep("^x\\.", names(d)), drop = FALSE]
  out <- cbind(all, params)
  body <- c(paste(names(out), collapse = ","),
            apply(out, 1, function(r) paste(format(r, digits = 10, trim = TRUE), collapse = ",")))
  c(header, extra_comments, body)
}
min_hdr <- function(i) { h <- minimal[[i]][is_comment(minimal[[i]])]; h }
nuts_cols <- c("lp__", "accept_stat__", "stepsize__", "treedepth__", "n_leapfrog__", "divergent__", "energy__")
show <- function(name, r) {
  cat(sprintf("%-55s %s", name, if (isTRUE(r$ok)) "OK" else paste("FAIL:", r$why)))
  nw <- setdiff(r$warns, base_warns)
  if (length(nw)) cat("  [warnings:", paste(unique(nw), collapse = " | "), "]")
  cat("\n")
}
cat("\n== Probes\n")
show("P0 nuts-rs draws, minimal header, NUTS columns",
     probe(lapply(1:4, function(i) mk(i, min_hdr(i), nuts_cols)), "p0"))
prov <- c("# laplace_sampler = hybrid (./hybrid.lapsampler sha256:0123abcd)",
          "# laplace_kernel = beta: nuts() builtin laplace-sample 0.1.0",
          "# laplace_kernel = sigma,tau: langevin::mala 0.1.0 sha256:89ab")
r <- probe(lapply(1:4, function(i) mk(i, min_hdr(i), nuts_cols, prov)), "p1")
show("P1 + '# laplace_sampler/laplace_kernel = ...' lines", r)
if (isTRUE(r$ok)) cat("   metadata()$laplace_sampler =", r$fit$metadata()$laplace_sampler,
                      "| laplace_kernel =", r$fit$metadata()$laplace_kernel, "(last one wins)\n")
r <- probe(lapply(1:4, function(i) mk(i, min_hdr(i), c(nuts_cols, "n_grad__", "accept_stat_beta__"))), "p2")
show("P2 + extra columns n_grad__, accept_stat_beta__", r)
if (isTRUE(r$ok)) {
  cat("   sampler_diagnostics:", paste(dimnames(r$fit$sampler_diagnostics())$variable, collapse = ","), "\n")
  cat("   summary variables:  ", paste(r$s$variable, collapse = ","), "\n")
}
mala_cols <- c("lp__", "accept_stat__", "stepsize__", "n_grad__")
r <- probe(lapply(1:4, function(i) mk(i, min_hdr(i), mala_cols)), "p3")
show("P3 non-NUTS column set (lp, accept, stepsize, n_grad)", r)
if (isTRUE(r$ok)) { cat("   diagnostic_summary:\n"); str(r$ds) }
alg_hdr <- function(i, alg) sub("algorithm = hmc", paste("algorithm =", alg), min_hdr(i), fixed = TRUE)
show("P4 'algorithm = laplace' + non-NUTS columns",
     probe(lapply(1:4, function(i) mk(i, alg_hdr(i, "laplace"), mala_cols)), "p4"))
show("P5 'algorithm = laplace' + NUTS columns",
     probe(lapply(1:4, function(i) mk(i, alg_hdr(i, "laplace"), nuts_cols)), "p5"))
show("P6 no comment header at all",
     probe(lapply(1:4, function(i) mk(i, character(), nuts_cols)), "p6"))

# ---- 3. Write the working example (P1 + P2 shape).
dir.create("out/csv", showWarnings = FALSE)
for (i in 1:4) writeLines(mk(i, min_hdr(i), c(nuts_cols, "n_grad__", "accept_stat_beta__"), prov),
                          sprintf("out/csv/example-%d.csv", i))
ex <- as_cmdstan_fit(sprintf("out/csv/example-%d.csv", 1:4))
cat("\n== Example out/csv/example-*.csv summary\n"); print(ex$summary())
