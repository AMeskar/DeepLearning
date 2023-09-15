#!/bin/bash

directory="Files"  # Replace with the desired directory path

# Check if the directory exists
if [ -d "$directory" ]; then
    # List all files in the directory and print their names
    for file in "$directory"/*; do
        if [ -f "$file" ]; then		
            	sbatch -t 3-00:30 -n 1 --mem 10G  readtxt.sh $file
        fi
    done
else
    echo "Directory not found."
fi

echo $1
