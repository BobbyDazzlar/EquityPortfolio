#!/bin/bash
#PBS -l select=1:ncpus=12
#PBS -N PSO
#PBS -l software=python
#PBS -V
#PBS -q workq



/usr/bin/python3 /home/pn_kumar/Karthik/window-sliding/PSO.py

