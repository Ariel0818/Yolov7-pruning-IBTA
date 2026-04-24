"""
ioucal.py – IoU Calculator Utility

Standalone utility for computing Intersection over Union (IoU) between two
bounding boxes. Supports both standard corner format and centre format.
"""

import argparse
import numpy as np


def compute_iou(box1, box2, standard_coordinates=True):
    """
    Compute IoU between two bounding boxes.

    Args:
        box1: [Xmin, Ymin, Xmax, Ymax]  or  [Xcenter, Ycenter, W, H]
        box2: same format as box1
        standard_coordinates: True  -> [Xmin, Ymin, Xmax, Ymax]
                               False -> [Xcenter, Ycenter, W, H]

    Returns:
        float: IoU value in [0, 1]
    """
    if standard_coordinates:
        Xmin1, Ymin1, Xmax1, Ymax1 = box1
        Xmin2, Ymin2, Xmax2, Ymax2 = box2
    else:
        Xcenter1, Ycenter1, W1, H1 = box1
        Xcenter2, Ycenter2, W2, H2 = box2
        Xmin1 = int(Xcenter1 - W1 / 2)
        Ymin1 = int(Ycenter1 - H1 / 2)
        Xmax1 = int(Xcenter1 + W1 / 2)
        Ymax1 = int(Ycenter1 + H1 / 2)
        Xmin2 = int(Xcenter2 - W2 / 2)
        Ymin2 = int(Ycenter2 - H2 / 2)
        Xmax2 = int(Xcenter2 + W2 / 2)
        Ymax2 = int(Ycenter2 + H2 / 2)

    inter_Xmin = max(Xmin1, Xmin2)
    inter_Ymin = max(Ymin1, Ymin2)
    inter_Xmax = min(Xmax1, Xmax2)
    inter_Ymax = min(Ymax1, Ymax2)

    W = max(0, inter_Xmax - inter_Xmin)
    H = max(0, inter_Ymax - inter_Ymin)
    inter_area = W * H

    area1 = (Xmax1 - Xmin1) * (Ymax1 - Ymin1)
    area2 = (Xmax2 - Xmin2) * (Ymax2 - Ymin2)
    union_area = area1 + area2 - inter_area

    return inter_area / (union_area + 1e-6)


def parse_args():
    parser = argparse.ArgumentParser(description='Compute IoU between two bounding boxes')
    parser.add_argument('--box1', type=float, nargs=4, metavar=('X1', 'Y1', 'X2', 'Y2'),
                        required=True, help='First bounding box')
    parser.add_argument('--box2', type=float, nargs=4, metavar=('X1', 'Y1', 'X2', 'Y2'),
                        required=True, help='Second bounding box')
    parser.add_argument('--centre', action='store_true',
                        help='Input format is [Xcenter, Ycenter, W, H] instead of corner coords')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    iou = compute_iou(args.box1, args.box2, standard_coordinates=not args.centre)
    print(f"IoU: {iou:.4f}")
