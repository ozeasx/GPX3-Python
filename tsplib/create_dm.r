#! /usr/bin/Rscript

# Get script arguments
args = commandArgs(trailingOnly=TRUE)

# Load TSP package
library(TSP)

# Load TSPLIB file
tsp = read_TSPLIB(args)

# Create distance matrix
dm = dist(tsp)

# Convert distance matrix to pairwise distance matrix
# https://stackoverflow.com/questions/5813156/convert-and-save-distance-matrix-to-a-specific-format
#m = data.frame(t(combn(rownames(tsp),2)), as.numeric(dm))
m = as.numeric(dm)

# Write to file
# https://stackoverflow.com/questions/6750546/export-csv-without-col-names
write.table(m, paste(args, "dm", sep = '.'), quote = FALSE, row.names = FALSE,
            col.names = FALSE)
