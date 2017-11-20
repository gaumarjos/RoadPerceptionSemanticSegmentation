#!/bin/bash
# nohup python main.py train --gpu=1 --xla=2 -ep=1 -bs=10 -lr=0.0001 > nohup.out 2>&1 &
python main.py train --gpu=1 --xla=2 -ep=78 -bs=10 -lr=0.00001
