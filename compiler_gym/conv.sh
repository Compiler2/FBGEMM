#!/bin/bash

for((x=0;x<=11;x+=1))
do
  make conv CASE=$x
  ./conv
  make clean
done
