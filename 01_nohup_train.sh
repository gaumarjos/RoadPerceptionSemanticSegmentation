#!/bin/bash
nohup python main.py train --gpu=1 --xla=2 -ep=10 -bs=10 -lr=0.00001 > nohup.out 2>&1 &
