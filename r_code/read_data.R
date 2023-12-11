library(purrr)
library(dplyr)
library(data.table)


# Function to extract data from a single file
extract_data_from_file <- function(file, columns) {
  data <- fread(file,
                select = columns,
                header = TRUE,
                fill = TRUE,
                stringsAsFactors = FALSE)
  return(data)
}

load_multiple_sets <- function(db_paths, columns) {
        combined_data <- map_df(db_paths, function(x) extract_data_from_file(x, columns))
        return(combined_data)

}
