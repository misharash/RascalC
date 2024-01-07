#!/bin/bash
srun -N 1 -C cpu -t 00:05:00 --qos interactive python -u run_cov.py 4