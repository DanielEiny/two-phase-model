library(shazam)

source("r_code/read_data.R")


columns_list <- c("sequence_alignment", "ancestor_alignment", "v_call")
all_sets <- read.csv("data/final_sets.csv",
                     header = TRUE,
                     stringsAsFactors = FALSE)

paths <- all_sets$path
to_remove <- "/home/bcrlab/daniel/two-phase-model/data/P4/P4_I19_S1_cloned_w_filtered_seqs.tsv"
paths <- paths[-grep(to_remove, paths)]

# paths <- paths[1:2]  # TODO: remove this

dataset <- load_multiple_sets(paths, columns_list)
print(" --- dataset loaded --- ")
print(dim(dataset))

# Create substitution model using silent mutations
sub_model <- createSubstitutionMatrix(dataset,
                                      model = "s",
                                      sequenceColumn = "sequence_alignment",
                                      germlineColumn = "ancestor_alignment",
                                      vCallColumn = "v_call")
print(" --- substitution model ready --- ")

# Create mutability model using silent mutations
mut_model <- createMutabilityMatrix(dataset,
                                    sub_model,
                                    model = "s",
                                    sequenceColumn = "sequence_alignment",
                                    germlineColumn = "ancestor_alignment",
                                    vCallColumn = "v_call")
print(" --- targeting model ready --- ")

mut_model_table <- mut_model@.Data
names(mut_model_table) <- mut_model@names
write.csv(mut_model_table,  "results/s5f/mutability_all_data.csv")
