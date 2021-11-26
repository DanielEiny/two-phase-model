library(shazam)


calc_thresold <- function(sample_path) {
        subsample_size <- 1000
        repertoire <- read.table(sample_path,
                                 sep = "\t",
                                 header = TRUE,
                                 fill = TRUE)
        hist_ham <- distToNearest(repertoire,
                                  model = "ham",
                                  subsample = subsample_size,
                                  first = FALSE,
                                  nproc = 60)
        output <- findThreshold(hist_ham$dist_nearest,
                                   method = "gmm",
                                   model = "gamma-gamma")
        return(output@threshold)
}

args <- commandArgs(trailingOnly = TRUE)
calc_thresold(args[1])
