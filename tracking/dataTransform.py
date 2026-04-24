"""
dataTransform.py – Data Format Conversion Utilities

Converts between the three annotation formats used in this project:

  1. Raw GT format  (DarkLabel export):
        frame, id, class, x, y, w, h
  2. SORT input format  (MOT15-2D):
        frame, -1, x, y, w, h, 1, -1, -1, -1
  3. SORT result / evaluation format:
        frame, id, x, y, w, h, 1, -1, -1, -1
  4. DarkLabel visualisation format:
        frame, label, id, x, y, w, h

Usage examples
--------------
  # Convert GT to evaluation format (removes class column, appends MOT trailing cols)
  python dataTransform.py --mode gt2eval --input gt_data/flowerGT/4.txt --output Fgt4.txt

  # Convert GT to SORT input format (sets id=-1, removes class column)
  python dataTransform.py --mode gt2sort --input gt_data/flowerGT/4.txt --output flowerV-SORT/4.txt

  # Convert SORT result to DarkLabel visualisation format
  python dataTransform.py --mode sort2darklabel --input output/flower4.txt --output flower4_vis.txt --label "mature fruit"
"""

import argparse
import codecs
import numpy as np


# ---------------------------------------------------------------------------
# Conversion functions
# ---------------------------------------------------------------------------

def transform_gt_to_sort(filename):
    """GT -> SORT input: set id=-1, remove class column, append MOT trailing cols."""
    newdata = np.array([[1] * 10])
    with open(filename, 'r') as f:
        for line in f.readlines():
            array = line.strip('\n').split(',')
            array[1] = -1
            del array[2]           # remove class label
            array = np.append(array, [1, -1, -1, -1])
            newdata = np.concatenate((newdata, [array]), axis=0)
    return np.delete(newdata, 0, axis=0)


def transform_gt_to_eval(filename):
    """GT -> evaluation format: remove class column, append MOT trailing cols."""
    newdata = np.array([[1] * 10])
    with open(filename, 'r') as f:
        for line in f.readlines():
            array = line.strip('\n').split(',')
            del array[1]           # remove class label, keep id
            array = np.append(array, [1, -1, -1, -1])
            newdata = np.concatenate((newdata, [array]), axis=0)
    return np.delete(newdata, 0, axis=0)


def transform_sort_result_to_darklabel(filename, label='mature fruit'):
    """SORT result -> DarkLabel visualisation format."""
    out = np.array([[1] * 7])
    with open(filename, 'r') as f:
        for line in f.readlines():
            array = line.strip('\n').split(',')
            row = [array[0], label] + array[1:6]
            out = np.concatenate((out, [row]), axis=0)
    return np.delete(out, 0, axis=0)


def save(newdata, savename):
    """Write a numpy array to a CSV file."""
    with codecs.open(savename, 'w', 'utf-8') as f:
        for i in range(newdata.shape[0]):
            f.write(','.join(str(newdata[i, j]) for j in range(newdata.shape[1])))
            f.write('\r\n')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description='Convert between MOT annotation formats')
    parser.add_argument('--mode', type=str, required=True,
                        choices=['gt2sort', 'gt2eval', 'sort2darklabel'],
                        help='Conversion mode')
    parser.add_argument('--input', type=str, required=True, help='Input file path')
    parser.add_argument('--output', type=str, required=True, help='Output file path')
    parser.add_argument('--label', type=str, default='mature fruit',
                        help='Class label string (used in sort2darklabel mode)')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    if args.mode == 'gt2sort':
        data = transform_gt_to_sort(args.input)
    elif args.mode == 'gt2eval':
        data = transform_gt_to_eval(args.input)
    elif args.mode == 'sort2darklabel':
        data = transform_sort_result_to_darklabel(args.input, label=args.label)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    save(data, args.output)
    print(f"Saved {len(data)} rows to {args.output}")
