"""
auto_calibrate.py

Self-calibrating functions for robust parking slot detection:
- Auto-Canny thresholds based on image brightness
- Auto-Hough threshold based on edge density
- Auto-tuning of sensitive parameters with scoring
"""

import numpy as np
import cv2


def auto_canny(gray, sigma=0.33):
    """
    Automatically set Canny thresholds based on median pixel intensity.
    
    Args:
        gray: Grayscale image
        sigma: Control parameter (0.33 is typical)
    
    Returns:
        (edges, (low_threshold, high_threshold))
    """
    v = np.median(gray)
    lo = int(max(0, (1.0 - sigma) * v))
    hi = int(min(255, (1.0 + sigma) * v))
    edges = cv2.Canny(gray, lo, hi)
    return edges, (lo, hi)


def auto_hough_threshold(edges):
    """
    Automatically set Hough threshold based on edge density.
    
    Args:
        edges: Binary edge image from Canny
    
    Returns:
        threshold: Recommended Hough threshold value
    """
    edge_density = np.count_nonzero(edges) / edges.size
    
    # Tune based on edge density
    if edge_density < 0.01:
        return 10
    if edge_density < 0.03:
        return 20
    if edge_density < 0.06:
        return 35
    return 60


def score_detection(segments, roi_shape):
    """
    Score a lane detection result to find the best parameter set.
    
    Higher score = better detection (two parallel lines, good length, clean)
    
    Args:
        segments: List of detected line segments [(x1,y1,x2,y2), ...]
        roi_shape: (H, W) shape of ROI
    
    Returns:
        score: Float score (higher is better, negative if invalid)
    """
    if len(segments) < 2:
        return -1e9

    H, W = roi_shape[:2]

    # Compute segment properties
    lengths = []
    angles = []
    xs = []
    for (x1, y1, x2, y2) in segments:
        lengths.append(np.hypot(x2 - x1, y2 - y1))
        angles.append(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        xs.append((x1 + x2) / 2.0)

    total_len = float(np.sum(lengths))
    
    # Must have sufficient line content
    if total_len < 0.6 * H:
        return -1e9

    # Penalize too many lines (noise)
    noise_pen = max(0, len(segments) - 8) * 50.0

    # Reward angle consistency (low std = parallel lines)
    ang_std = float(np.std(np.array(angles)))
    ang_reward = 200.0 / (1.0 + ang_std)

    # Reward good separation into two sides by x distribution
    xs_array = np.array(xs)
    spread = float(np.std(xs_array))
    spread_reward = 0.0
    if 0.10 * W < spread < 0.45 * W:
        spread_reward = 300.0

    score = total_len + ang_reward + spread_reward - noise_pen
    
    return score


def auto_tune(detector_class, edges, base_params, verbose=False):
    """
    Automatically tune sensitive parameters by grid search with scoring.
    
    Args:
        detector_class: HoughLaneDetector class
        edges: Binary edge image
        base_params: Base parameter dictionary
        verbose: Print search progress
    
    Returns:
        (best_params, best_line_img, best_segments, best_score)
    """
    best = None
    best_score = -1e18
    roi_shape = edges.shape

    # Grid search over sensitive parameters
    threshold_candidates = [10, 20, 35, 60]
    angle_pair_th_candidates = [4, 6, 8, 12, 16]
    
    total_trials = len(threshold_candidates) * len(angle_pair_th_candidates)
    trial_num = 0

    for thr in threshold_candidates:
        for apt in angle_pair_th_candidates:
            trial_num += 1
            
            # Create candidate parameters
            params = dict(base_params)
            params["threshold"] = thr
            params["angle_pair_th"] = float(apt)
            
            # Disable auto-scale during grid search to use explicit values
            params["auto_scale"] = False

            # Test this parameter set
            det = detector_class(params)
            line_img, segs = det.detect(edges)
            s = score_detection(segs, roi_shape)

            if verbose:
                print(f"  Trial {trial_num}/{total_trials}: thr={thr}, apt={apt} → {len(segs)} segs, score={s:.1f}")

            if s > best_score:
                best_score = s
                best = (params, line_img, segs, best_score)

    if verbose and best is not None:
        print(f"\nBest: threshold={best[0]['threshold']}, angle_pair_th={best[0]['angle_pair_th']}, score={best_score:.1f}")

    return best if best is not None else (base_params, None, [], -1e18)
