## Script to collect the raw covariance matrices from all subdirectories

import os, argparse
from glob import glob
from RascalC.raw_covariance_matrices import collect_raw_covariance_matrices


def process_dir(dirname):
    print("Entered", dirname)
    if glob("CovMatricesAll/c4*_0*.txt", root_dir=dirname): # simple heuristic for existing raw covariance matrix text files to be collected
        collect_raw_covariance_matrices(dirname, check_finished=args.check_finished)
    for filename in os.listdir(dirname):
        filepath = os.path.join(dirname, filename)
        if os.path.isdir(filepath):
            process_dir(filepath)


parser = argparse.ArgumentParser(description="Collect raw covariance matrices from all subdirectories")
parser.add_argument("root_dir", help="Root directory under which to search for raw covariance matrix files")
parser.add_argument("--check_finished", action="store_true", help="Perform safety checks for unfinished runs (default: False, assuming that the jobs have finished long ago)")
args = parser.parse_args()

process_dir(args.root_dir)