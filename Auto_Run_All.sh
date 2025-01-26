#!/bin/bash

directory="Files"  

if [ -d "$directory" ]; then

    for file in "$directory"/*; do
        if [ -f "$file" ]; then		
            	sbatch -t 3-00:30 -n 1 --mem 10G  readtxt.sh $file
        fi
    done
else
    echo "Directory not found."
fi

echo $1
