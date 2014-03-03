#!/bin/sh

if [ -f output/cudamatrix.txt ] && [ -f output/cpumatrix.txt ]; then
	result=$(diff output/cudamatrix.txt output/cpumatrix.txt -q)
	if [ -z "$result" ]; then
		echo "Output are the same"
	else
		echo $result
	fi
else
	echo "Please run \"make run\" first"
fi
