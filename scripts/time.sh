#!/bin/bash

START=1024
STEP=1024
TIMEOUT=60000
MAXSIZE=17408
AVGNUM=5

if [ $# -lt 1 ]; then
	echo "Usage: time.sh [APP]"
	echo "To time the specified implementation"
	exit
fi

time=0
rm $1.log

echo "Timing $1"

for ((x=$START; ( x <= $MAXSIZE ) && ( $(echo $time|awk '{print int($1)}') <= $TIMEOUT ); x += $STEP )); do
	scripts/generate-testfile.sh $x
	time=0
	echo "Data magnitude $x"
	for ((i=0; i < $AVGNUM ; i++)) do
		echo $i
		time=$($1 output/cudamatrix.txt input/matrix.txt | awk "{print \$1 + $time}")
	done
	echo "Total time is $time for $AVGNUM times of size $x"
	time=$(echo $time $AVGNUM | awk '{print $1/$2}')
	echo "Average time is $time"
	echo "$x $time" >> $1.log
done
