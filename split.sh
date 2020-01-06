# Usage: split.sh <original file> <split size, eg 70> <output file 1> <output file 2>
split -l $[ $(wc -l $1|cut -d" " -f1) * $2 / 100 ] $1
mv xaa $3
mv xab $4

