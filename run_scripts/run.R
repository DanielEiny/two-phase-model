source("r_code/account_genealogy.R")


#samples_table <- read.csv("data/final_sets.csv")
#paths <- samples_table[samples_table$sample_id == "P4_I1_S1", "path"]
#paths <- samples_table[samples_table$study == "mg", "path"]

paths <- c("data/P4/P4_I15_S1_cloned_w_filtered_seqs.tsv",
           "data/P4/P4_I16_S1_cloned_w_filtered_seqs.tsv",
           "data/P4/P4_I17_S1_cloned_w_filtered_seqs.tsv",
           "data/P4/P4_I18_S1_cloned_w_filtered_seqs.tsv",
           "data/P4/P4_I19_S1_cloned_w_filtered_seqs.tsv",
           "data/P4/P4_I21_S1_cloned_w_filtered_seqs.tsv",
           "data/P4/P4_I22_S1_cloned_w_filtered_seqs.tsv",
           "data/P4/P4_I23_S1_cloned_w_filtered_seqs.tsv",
           "data/P4/P4_I24_S1_cloned_w_filtered_seqs.tsv",
           "data/P4/P4_I25_S1_cloned_w_filtered_seqs.tsv",
           "data/P4/P4_I26_S1_cloned_w_filtered_seqs.tsv",
           "data/P4/P4_I27_S1_cloned_w_filtered_seqs.tsv",
           "data/P4/P4_I28_S1_cloned_w_filtered_seqs.tsv",
           "data/P4/P4_I29_S1_cloned_w_filtered_seqs.tsv",
           "data/P4/P4_I30_S1_cloned_w_filtered_seqs.tsv",
           "data/P4/P4_I31_S1_cloned_w_filtered_seqs.tsv")

#lapply(paths[6:16], function(x) account_genealogy(x, x))


x <- "data/P3/P3_I2_S2_cloned_w_filtered_seqs.tsv"
account_genealogy(x, x)
