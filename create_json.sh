#!/bin/bash

wget https://raw.githubusercontent.com/json-iterator/test-data/master/large-file.json

tail -c +2 large-file.json > test1

truncate -s-1 test1

cat test1 >> test2
cat test2 >> test1
cat test1 >> test2
cat test2 >> test1
cat test1 >> test2
cat test2 >> test1
cat test1 >> test2
cat test2 >> test1
cat test1 >> test2
cat test2 >> test1
cat test1 >> test2

rm large-file.json test1
mv test2 test.json
