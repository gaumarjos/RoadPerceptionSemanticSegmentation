#!/bin/bash
nohup python main.py train --gpu=1 --xla=2 -ep=25 -bs=10 -lr=0.0001 > nohup.out 2>&1 &
