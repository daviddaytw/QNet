#!/bin/bash

REPO_FOLDER=$(dirname $(dirname $(readlink -f $0)))

IFS=','
while read DATASET MODEL EMBED_SIZE NUM_BLOCK
do
    $REPO_FOLDER/scripts/run-in-taiwania-1.sh -m $MODEL -d $DATASET -ed $EMBED_SIZE -nb $NUM_BLOCK
done < <(tail -n +2 $REPO_FOLDER/train_array.csv)
