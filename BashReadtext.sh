#!/bin/bash
pip3 install six --user
pip3 install ROOT --user

filename=$1
while IFS= read -r line
do
  echo $line
  rm -rf $line
  mkdir $line
  cd $line
  rfcp  /hpss/in2p3.fr/group/antares/mc/rbr/v4/km3/"$line" ./
  tar -xvf "$line"
  rm -rf *_a_*
  bzip2 -d *.bz2
        directory="../../Inputs/$line"
        for file in "$directory"/*.evt; do
            echo $file
            python3 ../test.py $file
        done
  cd ..
  rm -rf  $line
done < "$filename"
