#!/bin/bash
#PBS -l select=1:ncpus=12
#PBS -N fdata-MVO
#PBS -l software=python
#PBS -V
#PBS -q workq



/usr/bin/python3 /home/pn_kumar/Karthik/fdata-portfolio-optimization/fdata-MVO/fdata-MVO.py


