#!/bin/bash

THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,lib.cnmem=.70 python3 $1
