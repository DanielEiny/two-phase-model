library(igraph)
library(alakazam)
library(dplyr)
library(stringr)

source("r_code/list_of_columns_to_keep.R")

account_genealogy <- function(input_file_path, out_file_path) {
    # ----- Read & filter data ----- #
    repertoire <- read.table(input_file_path,
                             sep = "\t",
                             header = TRUE,
                             fill = TRUE)
    #repertoire <- repertoire[keep]
    #na_filter <- !is.na(repertoire$clone_id)
    #fake_filter <- !sapply(repertoire$sequence_id,
    #                       FUN = grepl,
    #                       pattern = "FAKE")
    #repertoire <- repertoire[na_filter & fake_filter, ]

    # Pad germline where sequence padded to fixed clone sequence length
    repertoire$germline_alignment <- str_pad(string = repertoire$germline_alignment,
                                             width = nchar(repertoire$sequence_alignment),
                                             side = "right",
                                             pad = "N")

    # ----- Preprocess clones ----- #
    clones <- repertoire %>%
              group_by(clone_id) %>%
              do(CHANGEO = makeChangeoClone(.,
                                            germ = "germline_alignment_d_mask",
                                            text_fields = c("c_call"),
                                            num_fields = "duplicate_count"))

    # ----- Reconstruct lineages ----- #
    phylip_exec <- "/home/bcrlab/daniel/two-phase-model/phylip-3.697/exe/dnapars"
    graphs <- lapply(clones$CHANGEO,
                     buildPhylipLineage,
                     phylip_exec = phylip_exec,
                     rm_temp = TRUE)
    graphs[sapply(graphs, is.null)] <- NULL  # In case of singleton clone

    # ----- Add new columns to data frame ----- #
    repertoire$ancestor_alignment <- repertoire$germline_alignment
    repertoire$sequence_origin <- "OBSERVED"
    repertoire$ancestor_origin <- "GERMLINE"

    # ----- Loop over lineage graphs, fill columns and append rows ----- #
    inferred_sequences_counter <- 0
    for (g in graphs){
            edges <- get.edgelist(g)
            vertex_attributes <- get.vertex.attribute(g)

            clone_representative_id <- vertex_attributes$label[which(!grepl("Inferred|Germline", vertex_attributes$label))[1]]
            clone_representative <- repertoire[which(repertoire$sequence_id == clone_representative_id), ]
            
            for (i in 1:dim(edges)[1]){
                    descendant_id <- edges[i, 2]
                    ancestor_id <- edges[i, 1]
                    ancestor_alignment <- vertex_attributes$sequence[which(vertex_attributes$label == ancestor_id)]

                    # Fill ancestor sequence / add entire row
                    observed <- !grepl("Inferred", descendant_id)

                    if (observed) {  # Observed sequence, row already exist
                            descendant_loc <- which(repertoire$sequence_id == descendant_id)
                            repertoire$ancestor_alignment[descendant_loc] <- ancestor_alignment 

                    } else {  # Inferred sequence, add row
                            inferred_sequences_counter = inferred_sequences_counter + 1
                            new_row <- clone_representative
                            new_row$sequence_id = paste0("INFERRED_", as.character(inferred_sequences_counter))
                            new_row$sequence_origin <- "PHYLOGENY_INFERRED"
                            new_row$sequence_alignment <- vertex_attributes$sequence[i]
                            new_row$ancestor_alignment <- ancestor_alignment

                            descendant_loc <- nrow(repertoire) + 1
                            repertoire <- rbind(repertoire, new_row)
                    }

                    # Fill ancestor origin
                    if (ancestor_id == "Germline") {
                            ancestor_origin <- "Germline"
                    } else {
                            if (grepl("Inferred", ancestor_id)) {
                                    ancestor_origin <- "PHYLOGENY_INFERRED"
                            } else {
                                    ancestor_origin <- "OBSERVED"
                            }
                    }
                    repertoire$ancestor_origin[descendant_loc] <- ancestor_origin
            }
    }

    # ----- Save to file ----- #
    write.table(repertoire, file = out_file_path, sep = "\t", row.names = FALSE)
}

account_genealogy("data/P10/P10_I3_S3_cloned_w_filtered_seqs.tsv",
                  "data/tmp.tsv")
