#!/bin/bash
i=$((0))
find ./dataset/3_genres/ -name "*.mp3\[*\]" | while read line
do 
	mv "$line" ${line}$i.mp3
	i=$((i + 1))
done


mkdir -p dataset/3_genres/test/pop
ls dataset/3_genres/pop | perl -MList::Util=shuffle -e 'print shuffle(<STDIN>);' | head -20 | sed "s/^/mv\ dataset\/3_genres\/pop\//" | sed "s/$/\ dataset\/3_genres\/test\/pop/" > move.sh
sh move.sh
mv dataset/3_genres/pop dataset/3_genres/train

mkdir dataset/3_genres/test/hiphop
ls dataset/3_genres/hiphop | perl -MList::Util=shuffle -e 'print shuffle(<STDIN>);' | head -20 | sed "s/^/mv\ dataset\/3_genres\/hiphop\//" | sed "s/$/\ dataset\/3_genres\/test\/hiphop/" > move.sh
sh move.sh
mv dataset/3_genres/hiphop dataset/3_genres/train

mkdir dataset/3_genres/test/rock
ls dataset/3_genres/rock | perl -MList::Util=shuffle -e 'print shuffle(<STDIN>);' | head -20 | sed "s/^/mv\ dataset\/3_genres\/rock\//" | sed "s/$/\ dataset\/3_genres\/test\/rock/" > move.sh
sh move.sh
mv dataset/3_genres/rock dataset/3_genres/train

rm move.sh
