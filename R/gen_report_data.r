#! /usr/bin/Rscript
# Load TSP package
library(TSP)
library(colorspace)
library(stringr)
library(plyr)

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
                   "t3: None", "t1f: True", "t2f: False", "t3f: False", "p: 100",
                   "M: random", "R: 1.0", "r: 0.5", "S: random", "e: 0",
                   "c: 1.0", "m: 0", "t: 2opt", "G: False", "o: None",
                   "E: True", "F: False", "i: None", "s: None")

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
    if (params[[i]][j] == "t2: False") {
      params[[i]][j] = ""
    }
    if (params[[i]][j] == "t1: True") {
      params[[i]][j] = "Test 1"
    }
    if (params[[i]][j] == "t1: False") {
      params[[i]][j] = ""
    }
    if (params[[i]][j] == "t3: True") {
      params[[i]][j] = "Test 2"
    }
    if (params[[i]][j] == "t3: False") {
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
params = lapply(params, paste, collapse = " + ")


# Consolidate fitness data
fitness = lapply(fitness_files, read.csv2, sep = ',', dec = '.',
                 header = FALSE)
fitness = lapply(fitness, rowMeans)

# Set y min and max
ymin = min(sapply(fitness, min))
ymax = max(sapply(fitness, max))
rangeMm <- function(x, M, m){(x-m)/(M-m)}

# Scale data to [0,1] interval
fitness = lapply(fitness, rangeMm, ymax, ymin)

ymin = 0
ymax = 1

# Plot fitness
# Change the palette to avoid yellow
col.pal <- palette()
col.pal[7] <- "purple"
palette(col.pal)

colors = c(1:n)
linetype = c(1:n)
plotchar = c(1:n)

# labely = (ymax + ymin)/2
labely = 1

fitness_plot_file = paste(args[1], "fitness.png", sep = '')
png(fitness_plot_file, width=1024, height=1024)
par(mar=c(4,6,4,4))
plot(fitness[[1]], type = 'n', xlab = "Generation", ylab = "Fitness",
     xlim = c(0, 1000), ylim = c(ymin, ymax), cex.axis=2, cex.lab=2)
for (i in 1:n) {
  lines(fitness[[i]], type = 'l', lty = linetype[i], col = colors[i],
        pch = plotchar[i], lwd=2)
  lines(fitness[[i]], type = 'p', lty = linetype[i], col = colors[i],
        pch = c(plotchar[i], rep(NA, 50)), cex = 2, lwd=2)
}
legend(10, labely, params, lty = linetype, col=colors, cex=2,
       pch = plotchar)
dev.off()

# Statistical tests

ttest <- function(x) {
  t.test(fitness[[x[1]]],fitness[[x[2]]])
}

combn(n,2, FUN= ttest, simplify = FALSE)


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
