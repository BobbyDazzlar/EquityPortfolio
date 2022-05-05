#!/bin/bash
#PBS -l select=3:ncpus=12
#PBS -N ACO
#PBS -l software=python
#PBS -V
#PBS -q workq



/usr/bin/python3 /home/pn_kumar/Karthik/staticlookback30/ACO.py

