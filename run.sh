#!/usr/bin/bash

# Current possibilities:
#   - gen for generation
#   - coco for downloading and preparing coco
#   - iqa for measuring the quality of generated images
#   - train

if [[ $1 == "gen" || ($1 == "coco" || ($1 == "iqa" || ($1 == "train" || ($1 == "create_dataset" || $1 == "plot_results") ))) ]]; then
    python3 "src/$1.py" ${@:2}
else
    echo "Unrecognized utility $1"
fi
