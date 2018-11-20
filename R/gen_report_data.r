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
range01 <- function(x){(x-min(x))/(-tour_length(best_known_tour, tsp)-min(x))}
range02 <- function(x){(x-min(x))/(max(x)-min(x))}

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

# Get parametrization
trim <- function (x) gsub("^\\s+|\\s+$", "", x)
default_params = c("k: 0", "P: False", "p: 100", "M: random", "r: 0", "e: 0",
                   "c: 0", "x: GPX", "m: 0", "g: 100", "n: 1", "o: True",
                   "f1: True", "f2: True", "f3: False")

params = lapply(param_files, scan, sep = ',', what = "list")
params = lapply(params, trim)
params = lapply(params, setdiff, default_params)
params = lapply(params, sort)
params = lapply(params, paste, collapse = " ")

# Consolidate fitness data
fitness = lapply(fitness_files, read.csv2, sep = ',', dec = '.', header = FALSE)
fitness = lapply(fitness, rowMeans)
fitness = lapply(fitness, range01)

# Plot fitness
colors = rainbow(length(fitness))
linetype = c(1:length(fitness))
plotchar = seq(1:length(fitness))

fitness_plot_file = paste(args[1], "fitness.png", sep = '')
png(fitness_plot_file, width=1024, height=1024)
plot(fitness[[1]], type = 'n', xlab = "Generation", ylab = "Fitness")
for (i in 1:length(fitness)) {
  lines(fitness[[i]], type = 'l', lty = linetype[i],
        col = colors[i])
}
legend(10, 0.5, params, lty = linetype, col=colors)
dev.off()

plot_data = function(data_files, data_names, plot_file, yl) {
  data = lapply(data_files, read.csv2, sep = ',', dec = '.',
                header = FALSE, col.names = data_names)
  data = lapply(data, colMeans)
  data = lapply(data, scale, center = FALSE)

  # Plot data
  data_plot_file = paste(args[1], plot_file, sep = '')
  png(data_plot_file, width=1024, height=1024)
  plot(data[[1]], type = 'n', xaxt='n', xlab = "", ylab = yl)
  for (i in 1:length(data)) {
    lines(data[[i]], type = 'o', lty = linetype[i],
          col = colors[i], pch = plotchar[i])
  }
  axis(1, at=1:length(data_names), labels = data_names)
  legend(2, max(data[[1]])/2, params, lty = linetype, col = colors)
  dev.off()
}

# Counter data
counters_names = c("Cross", "Failed", "Improvement", "Feasible 1",
                   "Feasible 2", "Feasible 3", "Infeasible", "Fusions",
                   "Unsolved", "Infeasible Tours", "Mutations")

# Timers data
timers_names = c("Total", "Pop", "Eval", "Tournament",
                 "Recomb", "Part", "Simple Graph",
                 "Class", "Fusion", "Build", "Mutation",
                 "Pop Restart")

plot_data(counters_files, counters_names, "counters.png", "Counting")
plot_data(timers_files, timers_names, "timers.png", "Time (s)")
