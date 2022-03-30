#!/bin/bash
#PBS -l select=1:ncpus=12
#PBS -N MVO
#PBS -l software=python
#PBS -V
#PBS -q workq



/usr/bin/python3 /home/pn_kumar/Karthik/window-sliding/MVO.py

