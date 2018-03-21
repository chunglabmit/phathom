"""score.py - score ground-truth vs detected"""

import argparse
import numpy as np
import json
from scipy.optimize import linear_sum_assignment
from scipy.spatial.kdtree import KDTree


def precision(n_true_positive, n_false_positive, n_false_negative):
    """Compute precision from true/false positive/negative

    "precision is the fraction of relevant instances among the retrieved
    instances" (https://en.wikipedia.org/wiki/Precision_and_recall)

    :param n_true_positive: # of true positives in sample
    :param n_false_positive: # of false positives in sample
    :param n_false_negative: # of false negatives in sample
    :returns:  true positives / (true positives + false positives)
    """
    try:
        return float(n_true_positive) / (n_true_positive + n_false_positive)
    except ZeroDivisionError:
        return np.nan


def recall(n_true_positive, n_false_positive, n_false_negative):
    """Compute recall from true/false positive/negative

    "recall  is the fraction of relevant instances that have been retrieved
    over the total amount of relevant instances"
    (https://en.wikipedia.org/wiki/Precision_and_recall)

    :param n_true_positive: # of true positives in sample
    :param n_false_positive: # of false positives in sample
    :param n_false_negative: # of false negatives in sample
    :returns:  true positives / (true positives + false negatives)
    """
    try:
        return float(n_true_positive) / (n_true_positive + n_false_negative)
    except ZeroDivisionError:
        return np.nan


def f_score(n_true_positive, n_false_positive, n_false_negative):
    """Compute f-score (f-measure) from true/false positive/negative

    :param n_true_positive: # of true positives in sample
    :param n_false_positive: # of false positives in sample
    :param n_false_negative: # of false negatives in sample
    :returns:  the harmonic mean of precision and recall
    """
    p = precision(n_true_positive, n_false_positive, n_false_negative)
    r = recall(n_true_positive, n_false_positive, n_false_negative)
    return 2 * p * r / (p + r)

def match_centroids(c1, c2, max_distance, inf=100000.):
    """Find the best matching of centroids in c1 to centroids in c2

    Match centroids in c1 to those in c2, minimizing total distance between
    pairs with the constraint that no match is further away than max_distance.

    :param c1: an N1xM array of centroid coordinates (M is the dimension
               of the volume).
    :param c2: another N2xM array of centroid coordinates
    :param max_distance: the maximum allowed distance between pairs
    :param inf: a ridiculously large distance to use in place of true infinity
    :returns: two arrays - one with the index of the matching centroid in c2
           for each c1 and one with the index of the matching centroid in c1
           for each c2. An index of -1 indicates that no match was found.
    """
    #
    # The matrix consists of rows of c1 and alternatives for c2
    # and columns for c2 and alternatives for c1.
    #
    matrix = np.ones((len(c1) + len(c2), len(c2) + len(c1))) * inf
    #
    # Compile pairs less than the max distance
    #
    kdtree = KDTree(c1)
    c2_matches = kdtree.query_ball_point(c2, max_distance)
    for c2_idx, c1s in enumerate(c2_matches):
        if len(c1s) == 0:
            continue
        d = np.sqrt(np.sum(
            (c1[np.array(c1s)] - c2[c2_idx][np.newaxis, :]) ** 2, 1))
        for c1_idx, dd in zip(c1s, d):
            matrix[c1_idx, c2_idx] = dd * 2
    #
    # Connect c1 to its alternative
    #
    matrix[np.arange(len(c1)), np.arange(len(c1)) + len(c2)] = max_distance
    #
    # Connect c2 to its alternative
    #
    matrix[np.arange(len(c2)) + len(c1), np.arange(len(c2))] = max_distance
    #
    # There is no penalty for connecting alternatives to each other whatever
    # way can be done.
    #
    matrix[len(c1):, len(c2):] = 0
    #
    # Run munkres algorithm to do assignment
    #
    c1_result, c2_result = linear_sum_assignment(matrix)
    #
    # The return values: initially -1
    #
    c1_idxs = -np.ones(len(c1), np.int32)
    c2_idxs = -np.ones(len(c2), np.int32)
    mask = (c1_result < len(c1)) & (c2_result < len(c2))
    c1_idxs[c1_result[mask]] = c2_result[mask]
    c2_idxs[c2_result[mask]] = c1_result[mask]
    return c1_idxs, c2_idxs


def score_centroids(c_detected, c_gt, max_distance):
    """Compute precision/recall stats on centroids

    Find the best match of detected to ground-truth and then compute
    precision, recall and f_score from those.

    :param c_detected: an N1xM array of the detected centroids
    :param c_gt: an N2xM array of the ground-truth centroids
    :param max_distance: maximum allowed distance of a match
    :returns: a CentroidsScore with the following attributes:
        gt_per_detected - an array of the indices of the ground-truth match
              for each detected centroid. An index of -1 indicates that there
              was no match (false positive)
        detected_per_gt - an array of the indices of the detected match for
              each ground-truth centroid. An index of -1 indicates that there
              was no match (false negative)
        precision - the precision of matching - # truly detected / # detected
        recall - the recall of matching # truly detected / # in ground truth
        f_score - the f-score
    """
    d_idxs, gt_idxs = match_centroids(c_detected, c_gt, max_distance)
    n_tp = np.sum(d_idxs >= 0)
    n_fp = np.sum(d_idxs < 0)
    n_fn = np.sum(gt_idxs < 0)
    p = precision(n_tp, n_fp, n_fn)
    r = recall(n_tp, n_fp, n_fn)
    f = f_score(n_tp, n_fp, n_fn)

    class CentroidsScore:
        gt_per_detected = d_idxs
        detected_per_gt = gt_idxs
        precision = p
        recall = r
        f_score = f

    return CentroidsScore()


def parse_into_array(path):
    """Parse either a numpy or json-format array
    :param path: path to array saved using either numpy.save or json.dump
    """
    if path.endswith(".npy"):
        return np.load(path)
    return np.array(json.load(open(path)))


def main():
    """Compute f-score, precision and recall from the command-line
    """
    parser = argparse.ArgumentParser(
        epilog="Ground-truth and detected files can be an Nx3 array "
               "saved using numpy.save (use extension .npy) or json.dump "
               "(use extension .json).")
    parser.add_argument("--gt-path",
                        help="Path to list of ground-truth points",
                        required=True)
    parser.add_argument("--detected-path",
                        help="Path to list of detected points",
                        required=True)
    parser.add_argument("--max-distance",
                        help="Maximum distance between pairs of points",
                        required=True,
                        type=float)
    args = parser.parse_args()
    gt = parse_into_array(args.gt_path)
    detected = parse_into_array(args.detected_path)
    max_distance = args.max_distance
    score = score_centroids(detected, gt, max_distance)
    for key, value in (("precision", score.precision),
                       ("recall", score.recall),
                       ("f-score", score.f_score)):
        print("%12s %.3f" % (key, value))

if __name__=="__main__":
    main()