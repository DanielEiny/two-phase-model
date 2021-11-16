library(igraph)
library(alakazam)
library(dplyr)
library(stringr)


reconstruct_lineage_trees <- function(data_set_path) {
    # ----- Read & filter data ----- #
    # TODO: filter by conscount, dupcount - use cli arguments
    repertoire <- read.table(data_set_path,
                             sep = "\t",
                             header = TRUE,
                             fill = TRUE)
    na_filter <- !is.na(repertoire$clone_id)
    fake_filter <- !sapply(repertoire$sequence_id,
                           FUN = grepl,
                           pattern = "FAKE")
    repertoire <- repertoire[na_filter & fake_filter, ]

    # Pad germline where sequence padded to fixed clone sequence length
    repertoire$germline_alignment <- str_pad(string = repertoire$germline_alignment,
                                             width = nchar(repertoire$sequence_alignment),
                                             side = "right",
                                             pad = "N")

    # ----- Preprocess clones ----- #
    clones <- repertoire %>%
              group_by(clone_id) %>%
              do(CHANGEO = makeChangeoClone(.,
                                            text_fields = c("c_call"),
                                            num_fields = "duplicate_count"))

    # ----- Build lineages ----- #
    phylip_exec <- "/home/bcrlab/daniel/two-phase-model/phylip-3.697/exe/dnapars"
    graphs <- lapply(clones$CHANGEO,
                     buildPhylipLineage,
                     phylip_exec = phylip_exec,
                     rm_temp = TRUE)
    return(graphs)
}
