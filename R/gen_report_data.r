#! /usr/bin/Rscript

# Load TSP package
library(TSP)

# Get script arguments
args = commandArgs(trailingOnly=TRUE)

# Get path to TSP instance
report_file = paste(args[1], "report1.log", sep="")
instance_file = grep(".tsp", readLines(report_file), value=TRUE)
instance_file = strsplit(instance_file, " ")[[1]][3]

# Load TSP instance
tsp = read_TSPLIB(instance_file)

# Load best known tour
best_known_tour_file = paste(args[1], "best_known_tour.out", sep = '')
best_known_tour = TOUR(scan(best_known_tour_file, sep = ','))
# https://stackoverflow.com/questions/5665599/range-standardization-0-to-1-in-r
range01 <- function(x){(x-min(x))/(tour_length(best_known_tour, tsp)-min(x))}

# Generate various file paths
param_files = c()
fitness_files = c()
counters_files = c()
timers_files = c()
best_tour_files = c()

for (i in 1:length(args)) {
  param_files[i] = paste(args[i], "parametrization.out", sep="")
  fitness_files[i] = paste(args[i], "best_fitness.out", sep="")
  counters_files[i] = paste(args[i], "counters.out", sep="")
  timers_files[i] = paste(args[i], "timers.out", sep="")
  best_tour_files[i] = paste(args[i], "best_tour_found.out", sep="")
}

# Choose best tour found
for (i in 1:length(best_tour_files)) {
  tour = TOUR(scan(best_tour_files[i], sep = ','))
  if (!exists("best_tour")) {
    best_tour = tour
  } else if (tour_length(tour, tsp) < tour_length(best_tour, tsp)) {
    best_tour = tour
  }
}


# Plot best_tour found
tour_plot_file = paste(args[1], "tours.png", sep = '')
png(tour_plot_file, width=1024, height=1024)
par(pty="s")
plot(tsp, best_tour, asp = 1)
dev.off()

# Consolidate fitness data
fitness = lapply(fitness_files, read.csv2, sep = ',', dec = '.', header = FALSE)
fitness = lapply(fitness, rowMeans)
fitness = lapply(fitness, range01)

# Get parametrization
params = lapply(param_files, read.csv2, sep = ',', header = FALSE)

# Plot all fitness
colors = rainbow(length(fitness))
linetype = c(1:length(fitness))
plotchar = seq(1:length(fitness))

fitness_plot_file = paste(args[1], "fitness.png", sep = '')
png(fitness_plot_file, width=1024, height=1024)
plot(fitness[[i]], type = 'n', xlab = "Generation", ylab = "Fitness")
for (i in 1:length(fitness)) {
  lines(fitness[[i]], type = 'o', lty = linetype[i], col = colors[i],
        pch = plotchar[i])
}
dev.off()
