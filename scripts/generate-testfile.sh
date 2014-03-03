#!/bin/bash

Default=256

if [ $# -lt 1 ]; then
  size=$Default
  echo "Using default size of $size"
else
  size=$1
  echo "Using size $size"
fi

make testgen
testgen/washall-test $size input/matrix.txt
