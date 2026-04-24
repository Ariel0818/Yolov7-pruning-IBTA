"""
stepcal.py – Step Calculator

Estimates the average per-frame pixel displacement of objects across consecutive
frames in a MOT-format annotation file. The result is used as the `step` parameter
in sortwithstep.py and Vsort.py to compensate for robot movement.
"""

import argparse
import numpy as np


def compute_step(filename):
    """
    Load a MOT-format file and compute per-frame pixel displacement for each
    tracked object that appears in consecutive frames.

    The displacement is measured along the x-axis (column 4 in the file).

    Returns:
        steps (list[int]): list of per-frame displacements across all objects
    """
    with open(filename, 'r') as f:
        newdata = np.array([[1, 1, 1, 1, 1, 1, 1]])
        for line in f.readlines():
            line = line.strip('\n')
            array = line.split(',')
            newdata = np.concatenate((newdata, [array]), axis=0)
    newdata = np.delete(newdata, 0, axis=0)

    steps = []
    for i in range(len(newdata) - 1):
        # Same object ID in adjacent rows implies consecutive-frame appearance
        if newdata[i][2] == newdata[i + 1][2]:
            displacement = int(newdata[i + 1][4]) - int(newdata[i][4])
            steps.append(displacement)
    return steps


def parse_args():
    parser = argparse.ArgumentParser(description='Compute average per-frame step from a MOT annotation file')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to MOT-format ground-truth file (e.g. gt_data/flowerGT/4.txt)')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    steps = compute_step(args.input)
    if len(steps) == 0:
        print("No consecutive object pairs found in the file.")
    else:
        avg = sum(steps) / len(steps)
        print(f"Step values: {steps}")
        print(f"Average step: {avg:.2f} pixels/frame")
        print(f"\nUse --step {avg:.2f} in sortwithstep.py or Vsort.py")
