#!/usr/bin/env bash

rm -rf $3
if spark-submit --driver-memory 8G kmeans.py $1 $2
then
    echo "SUCCESS"
else
    echo "FAILURE"
fi
