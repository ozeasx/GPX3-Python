#!/bin/sh
# https://blog.goo.ne.jp/u1low_cheap/e/fadae8e9052a248068ffdaa4682901b1
gnuplot <<EOF
set terminal postscript eps color enhanced 12;
set out '$(basename $1 .tsp).eps';
set grid x;
set grid y;
set size square;
set title '';
set nokey;
set style line 1 lt 1 lc rgb "royalblue";
plot "$1" using 3:2 every ::8 with linespoints linestyle 1 notitle;
EOF
