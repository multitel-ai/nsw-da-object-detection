#!/usr/bin/bash

if [[ $1 == "gen" || ($1 == "coco" || ( $1 == "flickr30k" || ($1 == "iqa" || ($1 == "train" || ( $1 == "iqa_paper" || ( $1 == "create_dataset" || ( $1 == "download" || ( $1 == "test" || $1 == "create_n_train" ) ) ) ) ) )  ) ) ]]; then
    python3 "src/$1.py" ${@:2}
else
    echo "Unrecognized utility $1"
fi


