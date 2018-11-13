#! /usr/bin/Rscript

# Get script arguments
args = commandArgs(trailingOnly=TRUE)

# Load TSP package
library(TSP)

# Generate various file paths
report_file = paste(args, "report1.log", sep="")
avg_fitness_file = paste(args, "avg_fitness.out", sep="")
best_fitness_file = paste(args, "best_fitness.out", sep="")
best_tour_file = paste(args, "best_tour.out", sep="")
counters_file = paste(args, "counters.out", sep="")
timers_file = paste(args, "timers.out", sep="")
tour_plot_file = paste(args, "best_tour.png", sep="")

# Get path to TSP instance
instance_file = grep(".tsp", readLines(report_file), value=TRUE)
instance_file = strsplit(instance_file, " ")[[1]][3]

# Load TSP instance
tsp = read_TSPLIB(instance_file)

# Load best tour
best_tour = TOUR(scan(best_tour_file, sep=","))

png(tour_plot_file)
par(pty="s")
plot(tsp, best_tour, asp=1)
dev.off()
