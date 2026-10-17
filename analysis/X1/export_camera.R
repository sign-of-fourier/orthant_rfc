# Export bayesm camera (CBC) to long CSV: resp, task, alt, chosen, 10 attribute columns.
# Usage: Rscript export_camera.R <out.csv>
suppressMessages(library(bayesm)); data(camera)
a <- commandArgs(trailingOnly = TRUE)
rows <- list(); i <- 0
for (n in seq_along(camera)) {
  X <- camera[[n]]$X; y <- camera[[n]]$y; T <- length(y)
  for (t in seq_len(T)) for (k in 1:5) {
    i <- i + 1
    rows[[i]] <- c(n - 1, t - 1, k - 1, as.integer(y[t] == k), X[(t - 1) * 5 + k, ])
  }
}
d <- do.call(rbind, rows)
colnames(d) <- c("resp", "task", "alt", "chosen", colnames(camera[[1]]$X))
write.csv(d, a[1], row.names = FALSE)
