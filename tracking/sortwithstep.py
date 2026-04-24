"""
    SORT with Step Compensation for Agricultural Robot Tracking
    Based on SORT: A Simple, Online and Realtime Tracker
    Copyright (C) 2016-2020 Alex Bewley alex@bewley.ai

    Extended with:
    - Step compensation for camera-on-robot scenarios (fruit detection from mobile platforms)
    - Overlap area (wh) as a secondary matching criterion alongside IoU

    Original SORT paper:
        Bewley et al., "Simple online and realtime tracking", ICIP 2016.
        https://arxiv.org/abs/1602.00763
"""
from __future__ import print_function

import os
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage import io

import glob
import time
import argparse
import decimal
from filterpy.kalman import KalmanFilter

np.random.seed(0)


def linear_assignment(cost_matrix):
    try:
        import lap
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[y[i], i] for i in x if i >= 0])
    except ImportError:
        from scipy.optimize import linear_sum_assignment
        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))


def iou_batch(bb_test, bb_gt):
    """
    Computes IoU between two sets of bboxes in [x1,y1,x2,y2] format.
    Returns (iou_matrix, overlap_area_matrix).
    """
    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)
    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
              + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)
    return o, wh


def convert_bbox_to_z(bbox):
    """[x1,y1,x2,y2] -> [cx,cy,area,aspect_ratio]"""
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.
    y = bbox[1] + h / 2.
    s = w * h
    r = w / float(h)
    return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x, score=None):
    """[cx,cy,area,aspect_ratio] -> [x1,y1,x2,y2]"""
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if score is None:
        return np.array([x[0] - w/2., x[1] - h/2., x[0] + w/2., x[1] + h/2.]).reshape((1, 4))
    else:
        return np.array([x[0] - w/2., x[1] - h/2., x[0] + w/2., x[1] + h/2., score]).reshape((1, 5))


class KalmanBoxTracker(object):
    """
    Tracks a single object with a 7-dim constant-velocity Kalman filter.
    State: [cx, cy, area, r, vx, vy, varea]; observation: [cx, cy, area, r]
    """
    count = 0

    def __init__(self, bbox):
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1]])
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0]])
        self.kf.R[2:, 2:] *= 10.
        self.kf.P[4:, 4:] *= 1000.
        self.kf.P *= 10.
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01
        self.kf.x[:4] = convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def update(self, bbox):
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox))

    def predict(self):
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        return convert_x_to_bbox(self.kf.x)


def associate_detections_to_trackers(detections, trackers, iou_threshold=0.2, step=314.57):
    """
    Match detections to existing trackers via IoU with step compensation.

    Step compensation: the camera is mounted on a mobile robot, so fruit positions
    shift by a roughly constant pixel offset (`step`) each frame along the y-axis.
    Tracker bounding boxes are shifted by `step` before the IoU computation so
    that the match accounts for robot movement.

    A pair is also considered matched when the raw pixel overlap area exceeds 1000,
    even if IoU is below `iou_threshold` (handles large-but-densely-packed fruits).

    Returns: matches, unmatched_detections, unmatched_trackers
    """
    if len(trackers) == 0:
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)

    for i in range(len(trackers)):
        d = decimal.Decimal(trackers[i][0])
        decimal_places = -d.as_tuple().exponent
        if decimal_places > 6:
            s = str(d)
            # Freshly initialised tracker has many trailing zeros — needs compensation
            if int(s[6]) == 0 and int(s[7]) == 0 and int(s[8]) == 0:
                trackers[i][1] += step
                trackers[i][3] += step
            # Otherwise the tracker has already been updated by the Hungarian algorithm
        else:
            trackers[i][1] += step
            trackers[i][3] += step

    iou_matrix, wh = iou_batch(detections, trackers)

    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            matched_indices = linear_assignment(-iou_matrix)
    else:
        matched_indices = np.empty(shape=(0, 2))

    unmatched_detections = [d for d in range(len(detections)) if d not in matched_indices[:, 0]]
    unmatched_trackers = [t for t in range(len(trackers)) if t not in matched_indices[:, 1]]

    matches = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold and wh[m[0], m[1]] < 1000:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))

    matches = np.empty((0, 2), dtype=int) if len(matches) == 0 else np.concatenate(matches, axis=0)
    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


class Sort(object):
    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.2, step=314.57):
        """
        Args:
            max_age:       frames to keep a track alive without detections
            min_hits:      minimum detections before a track is reported
            iou_threshold: minimum IoU for detection-to-tracker match
            step:          pixel displacement per frame (robot movement compensation)
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0
        self.step = step

    def update(self, dets=np.empty((0, 5))):
        """
        Args:
            dets: numpy array [[x1,y1,x2,y2,score], ...]  (call once per frame)
        Returns:
            numpy array [[x1,y1,x2,y2,id], ...] for active tracks this frame
        """
        self.frame_count += 1

        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)

        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(
            dets, trks, self.iou_threshold, self.step)

        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :])

        for i in unmatched_dets:
            self.trackers.append(KalmanBoxTracker(dets[i, :]))

        ret = []
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            if (trk.time_since_update < 1) and (
                    trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d, [trk.id + 1])).reshape(1, -1))
            i -= 1
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)

        return np.concatenate(ret) if len(ret) > 0 else np.empty((0, 5))


def parse_args():
    parser = argparse.ArgumentParser(description='SORT with step compensation for robot-mounted cameras')
    parser.add_argument('--display', action='store_true', help='Display online tracker output (slow)')
    parser.add_argument('--seq_path', type=str, default='data000', help='Path to detections root')
    parser.add_argument('--phase', type=str, default='train', help='Sub-directory in seq_path')
    parser.add_argument('--max_age', type=int, default=2,
                        help='Max frames to keep a track alive without detections')
    parser.add_argument('--min_hits', type=int, default=0,
                        help='Min detections before a track is initialised')
    parser.add_argument('--iou_threshold', type=float, default=0.2, help='Minimum IoU for match')
    parser.add_argument('--step', type=float, default=314.57,
                        help='Pixel displacement per frame for robot movement compensation')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    total_time = 0.0
    total_frames = 0
    colours = np.random.rand(32, 3)

    if args.display:
        if not os.path.exists('mot_benchmark'):
            print('\n\tERROR: mot_benchmark symlink not found.\n')
            exit()
        plt.ion()
        fig = plt.figure()
        ax1 = fig.add_subplot(111, aspect='equal')

    if not os.path.exists('output'):
        os.makedirs('output')

    pattern = os.path.join(args.seq_path, args.phase, '*', 'det', 'det.txt')
    for seq_dets_fn in glob.glob(pattern):
        KalmanBoxTracker.count = 0
        mot_tracker = Sort(max_age=args.max_age, min_hits=args.min_hits,
                           iou_threshold=args.iou_threshold, step=args.step)
        seq_dets = np.loadtxt(seq_dets_fn, delimiter=',')
        seq = seq_dets_fn[pattern.find('*'):].split(os.path.sep)[0]

        with open(os.path.join('output', '%s.txt' % seq), 'w') as out_file:
            print("Processing %s." % seq)
            for frame in range(int(seq_dets[:, 0].max())):
                frame += 1
                dets = seq_dets[seq_dets[:, 0] == frame, 2:7]
                dets[:, 2:4] += dets[:, 0:2]  # [x,y,w,h] -> [x1,y1,x2,y2]
                total_frames += 1

                if args.display:
                    fn = os.path.join('mot_benchmark', args.phase, seq, 'img1', '%06d.jpg' % frame)
                    im = io.imread(fn)
                    ax1.imshow(im)
                    plt.title(seq + ' Tracked Targets')

                start_time = time.time()
                trackers = mot_tracker.update(dets)
                total_time += time.time() - start_time

                for d in trackers:
                    print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1' % (
                        frame, d[4], d[0], d[1], d[2] - d[0], d[3] - d[1]), file=out_file)
                    if args.display:
                        d = d.astype(np.int32)
                        ax1.add_patch(patches.Rectangle(
                            (d[0], d[1]), d[2] - d[0], d[3] - d[1],
                            fill=False, lw=3, ec=colours[d[4] % 32, :]))

                if args.display:
                    fig.canvas.flush_events()
                    plt.draw()
                    ax1.cla()

    print("Total Tracking took: %.3f seconds for %d frames or %.1f FPS" % (
        total_time, total_frames, total_frames / total_time))
