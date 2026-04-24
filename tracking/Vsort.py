"""
Vsort: Location-Score Assisted Tracker

A tracking method that augments IoU-based matching with a spatial
location-score heuristic. When two bounding boxes overlap, the larger
object is assigned score '0' and the smaller one is assigned '1' (left)
or '2' (right) based on its relative horizontal position. This score is
used as a fallback to resolve matches when IoU alone is insufficient.

Designed for fruit tracking on a mobile agricultural robot where the
camera moves at a roughly constant speed (`step` pixels per frame).
"""

import argparse
import numpy as np


def read(filename):
    """Load a MOT-format annotation file, dropping the confidence column."""
    newdata = np.array([[1, 1, 1, 1, 1, 1, 1]])
    with open(filename, 'r') as f:
        for line in f.readlines():
            line = line.strip('\n')
            array = line.split(',')
            newdata = np.concatenate((newdata, [array]), axis=0)
    newdata = np.delete(newdata, 0, axis=0)   # remove seed row
    newdata = np.delete(newdata, 2, axis=1)   # drop confidence column
    return newdata


def findnextframe(frameid, data):
    """Return all rows belonging to a given frame id."""
    det = np.array([[1, 1, 1, 1, 1, 1]])
    for i in range(len(data)):
        if data[i][0] == frameid:
            det = np.concatenate((det, [data[i]]), axis=0)
    det = np.delete(det, 0, axis=0)
    return det


def finddataindex(det, data):
    """Return the row index in `data` that matches `det` by frame and position."""
    for i in range(len(data)):
        if data[i][0] == det[0] and data[i][2] == det[2] and data[i][3] == det[3]:
            return i


def iou(det, trks, step):
    """
    Compute IoU between `det` and a step-shifted `trks` bounding box.
    Boxes are in [frame, id, x, y, w, h] format (top-left origin, y grows down).
    Returns (intersection_area, iou_score).
    """
    predict = trks.copy()
    predict[3] = int(predict[3]) + step
    detx1, detx2 = int(det[2]), int(det[2]) + int(det[4])
    dety1, dety2 = int(det[3]), int(det[3]) - int(det[5])
    trksx1, trksx2 = int(predict[2]), int(predict[2]) + int(predict[4])
    trksy1, trksy2 = int(predict[3]), int(predict[3]) - int(predict[5])
    xx1 = max(detx1, trksx1)
    yy1 = min(dety1, trksy1)
    xx2 = min(detx2, trksx2)
    yy2 = max(dety2, trksy2)
    wh = max(0, xx2 - xx1 + 1) * max(0, yy1 - yy2 + 1)
    union = int(det[5]) * int(det[4]) + int(predict[4]) * int(predict[5]) - wh
    o = wh / union if union > 0 else 0
    return wh, o


def rank(trks):
    """Sort bounding boxes by area (largest first)."""
    if len(trks) == 0:
        return trks
    areas = np.array([int(t[4]) * int(t[5]) for t in trks])
    order = np.argsort(-areas)
    return trks[order]


def location_score(trks):
    """
    Assign a location score to each bounding box:
      'none' – does not significantly overlap any other box
      '0'    – the larger box in an overlapping pair
      '1'    – the smaller box, located to the LEFT of the larger one
      '2'    – the smaller box, located to the RIGHT of the larger one
    """
    trks = rank(trks)
    trks = trks.tolist()
    trks = [item + [''] for item in trks]

    if len(trks) == 0:
        return np.array(trks)
    if len(trks) == 1:
        trks[0][-1] = 'none'
        return np.array(trks)

    for i in range(len(trks) - 1):
        if trks[i][-1] != '':
            continue
        for j in range(i + 1, len(trks)):
            _, iou_val = iou(trks[i], trks[j], 0)
            if iou_val == 0 and j == len(trks) - 1 and trks[i][-1] == '':
                trks[i][-1] = 'none'
            elif iou_val == 0:
                continue
            else:
                trks[i][-1] = '0'
                trks[j][-1] = '1' if trks[j][2] < trks[i][2] else '2'
                break

    if trks[-1][-1] == '':
        trks[-1][-1] = 'none'
    return np.array(trks)


def parse_args():
    parser = argparse.ArgumentParser(description='Vsort: location-score assisted fruit tracker')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to MOT-format ground-truth or detection file')
    parser.add_argument('--output', type=str, default='stepmatchoutput/output.txt',
                        help='Path for the output tracking result file')
    parser.add_argument('--step', type=int, default=315,
                        help='Pixel displacement per frame (robot movement compensation)')
    parser.add_argument('--iou_threshold', type=float, default=0.2,
                        help='Minimum IoU to accept a match')
    parser.add_argument('--wh_threshold', type=float, default=1000,
                        help='Minimum pixel overlap area to accept a match when IoU is low')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    data = read(args.input)
    firstframe = data[0][0]
    finalframe = data[-1][0]

    # Assign initial IDs to the first frame
    trks0 = findnextframe(firstframe, data)
    for k in range(len(trks0)):
        data[k][1] = k + 1
        trks0[k][1] = k + 1

    result = location_score(trks0)
    num_frames = int(finalframe) - int(firstframe)

    for i in range(num_frames):
        trksid = str(i + int(firstframe))
        trks = findnextframe(trksid, data)
        detid = str(i + int(firstframe) + 1)
        det = findnextframe(detid, data)

        if len(trks) == 0 or len(det) == 0:
            continue

        trks = location_score(trks)
        det = location_score(det)

        for d in range(len(det)):
            matched = False
            for t in range(len(trks)):
                wh_val, o = iou(det[d], trks[t], args.step)
                if o > args.iou_threshold or wh_val > args.wh_threshold:
                    objectid = int(trks[t][1])
                    det[d][1] = objectid
                    dataid = finddataindex(det[d], data)
                    data[dataid][1] = objectid
                    result = np.concatenate((result, [det[d]]), axis=0)
                    matched = True
                    break
                elif det[d][-1] == trks[t][-1] and o > 0:
                    objectid = int(trks[t][1])
                    det[d][1] = objectid
                    dataid = finddataindex(det[d], data)
                    data[dataid][1] = objectid
                    result = np.concatenate((result, [det[d]]), axis=0)
                    matched = True
                    break

            if not matched:
                objectid = int(np.max(result[:, 1].astype(int))) + 1
                det[d][1] = str(objectid)
                dataid = finddataindex(det[d], data)
                data[dataid][1] = objectid
                result = np.concatenate((result, [det[d]]), axis=0)

    # Append MOT-format trailing columns
    final = result.copy()
    n = len(final)
    final = np.concatenate((final, np.ones((n, 1), dtype=int)), axis=1)
    final = np.concatenate((final, np.full((n, 3), -1)), axis=1)

    import os
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    np.savetxt(args.output, final, fmt='%s', delimiter=',')
    print(f"Saved tracking result to {args.output}")
