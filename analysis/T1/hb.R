# HB-MNL (Sawtooth CBC/HB proxy): bayesm::rhierMnlRwMixture, one normal component.
# Usage: Rscript hb.R <in.csv> <out_prefix> <R iterations> <keep> <seed>
# in.csv columns: resp, task, alt, chosen, x1..xP (alternatives in the same order every task).
# Writes <out_prefix>_beta.csv (posterior means over the second half of kept draws, one row per
# respondent in resp order) and <out_prefix>_loglike.csv (log-likelihood trace of kept draws).
suppressMessages(library(bayesm))
a <- commandArgs(trailingOnly = TRUE)
d <- read.csv(a[1]); R <- as.integer(a[3]); keep <- as.integer(a[4]); set.seed(as.integer(a[5]))
xcols <- grep("^x", names(d)); nvar <- length(xcols)
p <- max(d$alt) + 1
lgt <- lapply(split(d, d$resp), function(r) {
  r <- r[order(r$task, r$alt), ]
  ch <- r[r$chosen == 1, ]
  list(y = ch$alt + 1, X = as.matrix(r[, xcols]))
})
# Sawtooth-like prior: prior variance 1, prior degrees of freedom 5 (added to nvar)
nu <- nvar + 5
out <- rhierMnlRwMixture(Data = list(p = p, lgtdata = lgt),
                         Prior = list(ncomp = 1, nu = nu, V = nu * diag(nvar)),
                         Mcmc = list(R = R, keep = keep, nprint = 0))
nk <- dim(out$betadraw)[3]
bm <- apply(out$betadraw[, , (nk %/% 2 + 1):nk, drop = FALSE], c(1, 2), mean)
write.csv(bm, paste0(a[2], "_beta.csv"), row.names = FALSE)
write.csv(data.frame(loglike = out$loglike), paste0(a[2], "_loglike.csv"), row.names = FALSE)
