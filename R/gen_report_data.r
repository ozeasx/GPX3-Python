#! /usr/bin/Rscript
# Load TSP package
library(TSP)
library(colorspace)
library(stringr)

# Get script arguments
args = commandArgs(trailingOnly=TRUE)

# Get path to TSP instance
report_file = paste(args[1], "report1.log", sep="")
instance_file = grep(".tsp", readLines(report_file), value=TRUE)
instance_file = strsplit(instance_file, " ")[[1]][2]


# Args size
n = length(args)

# Load TSP instance
tsp = read_TSPLIB(instance_file)

# Load best known tour
best_known_tour_file = paste(args[n], "best_known_tour.out", sep = '')
best_known_tour = scan(best_known_tour_file, sep = ',')
# https://stackoverflow.com/questions/5665599/range-standardization-0-to-1-in-r
range0b <- function(x){(x-min(x))/(-tour_length(best_known_tour, tsp)-min(x))}
rangeMm <- function(x, M, m){(x-m)/(M-m)}

# Generate various file paths
param_files = c()
fitness_files = c()
counters_files = c()
timers_files = c()
best_tour_files = c()

for (i in 1:n) {
  param_files[i] = paste(args[i], "parametrization.out", sep="")
  fitness_files[i] = paste(args[i], "best_fitness.csv", sep="")
  counters_files[i] = paste(args[i], "counters.csv", sep="")
  timers_files[i] = paste(args[i], "timers.csv", sep="")
  best_tour_files[i] = paste(args[i], "best_tour_found.out", sep="")
}

# Choose best tour found
for (i in 1:n) {
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
plot(tsp, best_tour, asp = 1, xlab="x", ylab="y", main=tour_length(best_tour))
dev.off()

# Get parametrization
trim <- function (x) gsub("\\s+", " ", str_trim(x))
default_params = c("k: 3", "P: None", "K: None", "t1: None", "t2: None",
                   "t3: None", "t1f: None", "t2f: None", "t3f: None", "p: 100",
                   "M: random", "R: 1.0", "r: 0.5", "S: random", "e: 0",
                   "c: 1.0", "m: 0", "t: 2opt", "G: False", "o: None")

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

# Remove empty elements from params
remove <- function(x) {
  r = c("")
  x[! x %in% r]
}

params = lapply(params, remove)
params = lapply(params, paste, collapse = ", ")


# Consolidate fitness data
fitness = lapply(fitness_files, read.csv2, sep = ',', dec = '.',
                 header = FALSE)
fitness = lapply(fitness, rowMeans)
# fitness = lapply(fitness, rangeMm, max(sapply(fitness, max)),
#                                    min(sapply(fitness, min)))

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
     xlim = c(0, length(fitness[[1]])), ylim = c(min(sapply(fitness, min)),
                                                 max(sapply(fitness, max))))
for (i in 1:n) {
  lines(fitness[[i]], type = 'l', lty = linetype[i], col = colors[i])
}
legend(10, -2500, params,
       lty = linetype, col=colors)
dev.off()

# Summarize data

# https://stackoverflow.com/questions/31965975/truncate-but-not-round-in-r
truncate <- function(x, digit){
  digit = 10^digit
  trunc(x*digit)/digit
}

# https://stackoverflow.com/questions/30113039/summary-statistics-of-multiple-data-frames-within-a-list
my_summary <- function(x){
  mean = truncate(mean(x), 4)
  std = truncate(sd(x), 4)
  paste(mean, std, sep = ' ± ')
}

summarize = function(data_files, file_name){

  data = lapply(data_files, read.csv2, sep = ',', dec = '.', header = TRUE)
  data = lapply(data, sapply, my_summary)
  data = do.call(rbind, data)

  file_name = paste(args[1], file_name, sep = '')
  write.csv(data, file_name)

}

summarize(counters_files, "counters_summary.csv")
summarize(timers_files, "timers_summary.csv")
