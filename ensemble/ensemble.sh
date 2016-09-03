#! /usr/bin/zsh

# ensemble method to generate ensemble.csv
paste -d"," 48compare.log 32currentbest.log dropout.log wide.log | sed 's/,/ /g' | python ensemble.py > ensemble.csv

