#!/bin/bash
#PBS -l select=6:ncpus=12
#PBS -N ACO
#PBS -l software=python
#PBS -V
#PBS -q workq



/usr/bin/python3 /home/pn_kumar/Karthik/window-sliding/ACO.py

