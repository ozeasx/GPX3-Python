#! /usr/bin/Rscript

# Load TSP package
library(TSP)
library(colorspace)

# Get script arguments
args = commandArgs(trailingOnly=TRUE)

# Get path to TSP instance
report_file = paste(args[1], "report1.log", sep="")
instance_file = grep(".vrp", readLines(report_file), value=TRUE)
instance_file = strsplit(instance_file, " ")[[1]][3]

# Args size
n = length(args)

# Load TSP instance
tsp = read_TSPLIB(instance_file)

# Load best known tour
best_known_tour_file = paste(args[n], "best_known_tour.out", sep = '')
best_known_tour = scan(best_known_tour_file, sep = ',')
# https://stackoverflow.com/questions/5665599/range-standardization-0-to-1-in-r
# range0b <- function(x){(x-min(x))/(-tour_length(best_known_tour, tsp)-min(x))}
# rangeMm <- function(x, M, m){(x-m)/(M-m)}

# Generate various file paths
param_files = c()
fitness_files = c()
counters_files = c()
timers_files = c()
best_tour_files = c()

for (i in 1:n) {
  param_files[i] = paste(args[i], "parametrization.out", sep="")
  fitness_files[i] = paste(args[i], "best_fitness.out", sep="")
  counters_files[i] = paste(args[i], "counters.out", sep="")
  timers_files[i] = paste(args[i], "timers.out", sep="")
  best_tour_files[i] = paste(args[i], "best_tour_found.out", sep="")
}

# Choose best tour found
#for (i in 1:n) {
#  tour = scan(best_tour_files[i], sep = ',')
#  if (!exists("best_tour")) {
#    best_tour = tour
#  } else if (tour_length(tour, tsp) < tour_length(best_tour, tsp)) {
#    best_tour = tour
#  }
#}

# Plot best_tour found
#tour_plot_file = paste(args[1], "tours.png", sep = '')
#png(tour_plot_file, width=1024, height=1024)
#par(pty="s")
#plot(tsp, best_tour, asp = 1, xlab="x", ylab="y", main=tour_length(best_tour))
#dev.off()

# Get parametrization
trim <- function (x) gsub("^\\s+|\\s+$", "", x)
default_params = c("k: 0", "P: False", "p: 100", "M: random", "r: 0", "e: 0",
                   "c: 0", "x: GPX", "m: 0", "g: 100", "n: 1", "o: True",
                   "f1: True", "f2: True", "f3: False")

params = lapply(param_files, scan, sep = ',', what = "list")
params = lapply(params, trim)
params = lapply(params, setdiff, default_params)
params = lapply(params, sort)

for (i in 1:n) {
  for (j in 1:length(params[[i]])) {
    if (grepl("/", params[[i]][j], fixed=TRUE)) {
      params[[i]][j] = ""
    }
    if (grepl("g:", params[[i]][j], fixed=TRUE)) {
      params[[i]][j] = ""
    }
    if (grepl("n:", params[[i]][j], fixed=TRUE)) {
      params[[i]][j] = ""
    }
  }
}

params = lapply(params, paste, collapse = " ")

# Consolidate fitness data
fitness = lapply(fitness_files, read.csv2, sep = ',', dec = '.', header = FALSE)
fitness = lapply(fitness, rowMeans)
fitness = lapply(fitness, range0b)

# Plot fitness
# Change the palette to avoid yellow
col.pal <- palette()
col.pal[7] <- "purple"
palette(col.pal)

colors = c(1:n)
linetype = c(1:n)
plotchar = seq(1:n)

fitness_plot_file = paste(args[1], "fitness.png", sep = '')
png(fitness_plot_file, width=1024, height=1024)
plot(fitness[[1]], type = 'n', xlab = "Generation", ylab = "Fitness",
     xlim = c(0, length(fitness[[1]])), ylim = c(0, 1.1))
for (i in 1:n) {
  lines(fitness[[i]], type = 'l', lty = linetype[i], col = colors[i])
}
legend(10, 0.5, params, lty = linetype, col=colors)
dev.off()

# Plot data
plot_data = function(data_files, data_names, columns, plot_file, ylab) {
  data = lapply(data_files, read.csv2, sep = ',', dec = '.', header = FALSE,
                col.names = data_names)
  data = lapply(data, colMeans)
  data = do.call(rbind, data)

  # Data subseting
  data = data[,columns]
  names = data_names[columns]
  print(names)

  # Ploting
  data_plot_file = paste(args[1], plot_file, sep = '')
  png(data_plot_file, width=1024, height=1024)
  plot(data[1,], type='n', xaxt='n', xlab ="", ylab=ylab, ylim=c(0,max(data)))
  for (i in 1:n) {
    lines(data[i,], type = 'o', lty = linetype[i], col = colors[i],
          pch = plotchar[i])
  }
  axis(1, at=1:length(names), labels = names)
  legend(2, max(data)/2, params, lty = linetype, col=colors)
  dev.off()
}

# Counter data
counters_names = c("Cross", "Failed", "Improvement", "Feasible 1",
                   "Feasible 2", "Feasible 3", "Infeasible", "Fusions",
                   "Unsolved", "Infeasible Tours", "Mutations")

# Timers data
timers_names = c("Total", "Population", "Evaluation", "Tournament",
                 "Recombinations", "Partitioning", "Simple Graph",
                 "Classification", "Fusion", "Build", "Mutation",
                 "Pop Restart")

ga_counters = c(1, 2, 11)
gpx_counters = c(4, 5, 6, 7, 8, 9, 10)
improvement = c(3)

ga_timers = c(2, 3, 4, 5, 11, 12)
gpx_timers = c(6, 7, 8, 9, 10)


plot_data(counters_files, counters_names, ga_counters, "ga_counters.png",
          "Counting")
plot_data(counters_files, counters_names, gpx_counters, "gpx_counters.png",
          "Counting")
#plot_data(counters_files, counters_names, improvement, "improvement.png",
#          "Percentage")
plot_data(timers_files, timers_names, ga_timers, "ga_timers.png",
          "Time (s)")
plot_data(timers_files, timers_names, gpx_timers, "gpx_timers.png",
          "Time (s)")
