#!/bin/bash

for((x=100;x<=1000;x+=100))
do
  make GEMM_M=$x GEMM_N=$x GEMM_K=$x
  ./gemm
  make clean
done
