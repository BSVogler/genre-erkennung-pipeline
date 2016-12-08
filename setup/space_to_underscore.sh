#!/bin/bash

find -name "* *" -type f | rename 's/ /_/g'
find -name "*\(*" -type f | rename 's/\(/_/g'
find -name "*\)*" -type f | rename 's/\)/_/g'
find -name "*\&*" -type f | rename 's/\&/_/g'
