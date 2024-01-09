#!/bin/bash
srun -N 1 -C cpu -t 00:03:00 --qos interactive bash run_cov.sh