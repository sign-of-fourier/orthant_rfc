# MVP-Gibbs: bayesm::rmvpGibbs (data augmentation), own-price margins.
# Usage: Rscript gibbs.R <X.csv> <Y.csv> <out_prefix> <R iterations> <keep> <seed>
# X, Y: n x p (log price ratios, 0/1 picks), no header. Margins: item intercept + own log price
# (cross-price columns would make each iteration O(n p^2 k^2)). Identification per draw:
# beta_j / sqrt(Sigma_jj), R = Sigma scaled to unit diagonal; posterior means over the second
# half of kept draws. Writes <out_prefix>_c.csv, _b.csv (p), _R.csv (p x p), _trace.csv.
suppressMessages(library(bayesm))
a <- commandArgs(trailingOnly = TRUE)
X <- as.matrix(read.csv(a[1], header = FALSE)); Y <- as.matrix(read.csv(a[2], header = FALSE))
R <- as.integer(a[4]); keep <- as.integer(a[5]); set.seed(as.integer(a[6]))
n <- nrow(X); p <- ncol(X)
D <- matrix(0, n * p, 2 * p)
rows <- (rep(seq_len(n), each = p) - 1) * p + rep(seq_len(p), n)
D[cbind(rows, rep(seq_len(p), n))] <- 1
D[cbind(rows, p + rep(seq_len(p), n))] <- as.vector(t(X))
out <- rmvpGibbs(Data = list(p = p, y = as.vector(t(Y)), X = D),
                 Mcmc = list(R = R, keep = keep, nprint = 0))
nk <- nrow(out$betadraw); use <- (nk %/% 2 + 1):nk
cs <- matrix(0, length(use), p); bs <- cs; Rs <- matrix(0, p, p); tr <- numeric(nk)
for (q in seq_len(nk)) {
  S <- matrix(out$sigmadraw[q, ], p, p); d <- sqrt(diag(S))
  tr[q] <- mean((S / outer(d, d))[upper.tri(S)])
  if (q %in% use) {
    i <- q - use[1] + 1
    cs[i, ] <- out$betadraw[q, 1:p] / d; bs[i, ] <- out$betadraw[q, p + 1:p] / d
    Rs <- Rs + S / outer(d, d)
  }
}
write.csv(colMeans(cs), paste0(a[3], "_c.csv"), row.names = FALSE)
write.csv(colMeans(bs), paste0(a[3], "_b.csv"), row.names = FALSE)
write.csv(Rs / length(use), paste0(a[3], "_R.csv"), row.names = FALSE)
write.csv(data.frame(mean_corr = tr), paste0(a[3], "_trace.csv"), row.names = FALSE)
