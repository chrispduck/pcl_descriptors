#!/bin/bash

DATASET_DIR=./python/datasets
LEN_DATASET_DIR=${#DATASET_DIR}

echo $LEN_DATASET_DIR

OUTDIR=./descriptors/
mkdir $OUTDIR
find  $DATASET_DIR -print0 | while IFS= read -r -d '' file
do 
    extension="${file: -4}"
    # echo $extension
    # echo "$file"
    if [ $extension = ".pcd" ]
    then
        echo "PCD to process"
        OUTFILE="$OUTDIR${file:$LEN_DATASET_DIR +1}" 
        BASE_FILE=${OUTFILE%????} # descriptors/dataset_type/baseball
        echo "Base file: $BASE_FILE"

        PCD_IN=$file

        # Compute ESF
        ESF_OUT=${BASE_FILE}_esf.csv
        echo ./build/compute_esf $PCD_IN $ESF_OUT
        ./build/compute_esf $PCD_IN $ESF_OUT

        # Compute SHOT
        # KEYPOINTS_OUT="${BASE_FILE}_shot_keypoints"
        KEYPOINTS_OUT="_"
        SHOT_OUT="${BASE_FILE}_shot" # will append .csv automatically
        echo ./build/compute_shot $PCD_IN $KEYPOINTS_OUT $SHOT_OUT
        ./build/compute_shot $PCD_IN $KEYPOINTS_OUT $SHOT_OUT

    elif [ $extension = ".csv" ]
    then
        echo "ignoring csv"

    elif [ $extension = 'E.md' ]
    then
        echo "ignoring readme"
    else
        echo "mkdir here"
        FOLDER="$OUTDIR${file:$LEN_DATASET_DIR +1}"
        echo $FOLDER
        mkdir $FOLDER
    fi
done



