# HB binary logit per (block, product) with household-level product intercepts (Sawtooth CBC/HB proxy, as
# analysis/T1/hb.R): bayesm::rhierMnlRwMixture on 2-alternative choices (buy = product dummy, not buy = 0),
# one normal component, full covariance.
# Usage: Rscript hb_bin.R <in.bin> <n> <blocks> <J> <out.bin> <R iterations> <keep> <ndraws> <seed>
# in.bin: int8 array (n, blocks, J) in C order. out.bin: float64 betadraws of the last <ndraws> kept draws,
# array (ndraws, n, J) in C order; out.bin.loglike: float64 log-likelihood trace of kept draws.
suppressMessages(library(bayesm))
a <- commandArgs(trailingOnly = TRUE)
n <- as.integer(a[2]); nb <- as.integer(a[3]); J <- as.integer(a[4])
R <- as.integer(a[6]); keep <- as.integer(a[7]); nd <- as.integer(a[8]); set.seed(as.integer(a[9]))
y <- readBin(a[1], "integer", n * nb * J, size = 1)
Y <- aperm(array(y, c(J, nb, n)), c(3, 2, 1))  # (n, blocks, J)
X <- matrix(0, 2 * nb * J, J)
X[cbind(2 * (seq_len(nb * J)) - 1, rep(seq_len(J), nb))] <- 1  # row order: block-major, product within block
lgt <- lapply(seq_len(n), function(h) list(y = 2L - as.integer(t(Y[h, , ])), X = X))
nu <- J + 5
out <- rhierMnlRwMixture(Data = list(p = 2, lgtdata = lgt),
                         Prior = list(ncomp = 1, nu = nu, V = nu * diag(J)),
                         Mcmc = list(R = R, keep = keep, nprint = 0))
nk <- dim(out$betadraw)[3]
d <- out$betadraw[, , (nk - nd + 1):nk, drop = FALSE]  # (n, J, nd)
writeBin(as.vector(aperm(d, c(2, 1, 3))), a[5], size = 8)  # C order (nd, n, J)
writeBin(as.numeric(out$loglike), paste0(a[5], ".loglike"), size = 8)
