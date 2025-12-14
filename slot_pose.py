"""
slot_pose.py

Functions to convert detected lane segments into a parking slot pose:
- Cluster segments into two boundary lines (left/right)
- Fit clean lines to reduce jitter
- Compute slot center, orientation (theta), width, and length
- Generate slot box corners for visualization
"""

import numpy as np
import cv2


def seg_to_line_normal_form(seg):
    """
    Convert segment (x1, y1, x2, y2) to line in normal form ax+by+c=0.
    Returns (a, b, c, angle_deg) where (a, b) is unit normal.
    """
    x1, y1, x2, y2 = seg
    dx, dy = x2 - x1, y2 - y1
    ang = np.degrees(np.arctan2(dy, dx))

    a, b = dy, -dx
    n = np.hypot(a, b)
    if n < 1e-6:
        return None
    a /= n
    b /= n
    c = -(a * x1 + b * y1)
    return a, b, c, ang


def fit_line_through_points(points_xy):
    """
    Fit a line to points using cv2.fitLine.
    Returns line in normal form ax+by+c=0 (unit-normal) and direction angle (deg).
    """
    pts = np.array(points_xy, dtype=np.float32).reshape(-1, 1, 2)
    vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).flatten()

    # direction angle
    ang = np.degrees(np.arctan2(vy, vx))

    # convert to unit normal a*x + b*y + c = 0
    # direction (vx,vy) -> normal (a,b) = (vy,-vx)
    a, b = vy, -vx
    n = np.hypot(a, b)
    a /= n
    b /= n
    c = -(a * x0 + b * y0)
    return float(a), float(b), float(c), float(ang)


def cluster_into_two_boundaries(segments, angle_ref=None):
    """
    Cluster segments into two groups by their line offset c 
    (after enforcing consistent normal direction).
    
    Returns (line_left, line_right) each as (a, b, c, angle_deg).
    """
    lines = []
    for s in segments:
        out = seg_to_line_normal_form(s)
        if out is None:
            continue
        a, b, c, ang = out

        # make normals consistent so 'c' is comparable:
        # force a>=0 (or if a==0 force b>=0)
        if a < 0 or (abs(a) < 1e-6 and b < 0):
            a, b, c = -a, -b, -c
            ang = (ang + 180.0) if ang < 0 else (ang - 180.0)

        lines.append((a, b, c, ang, s))

    if len(lines) < 2:
        return None, None

    # Sort by c (offset). Two lane boundaries should form two clusters in c.
    lines.sort(key=lambda t: t[2])
    cs = np.array([t[2] for t in lines], dtype=np.float32)

    # Find the biggest gap in sorted c's -> split into two groups
    gaps = cs[1:] - cs[:-1]
    split_idx = int(np.argmax(gaps)) + 1

    group1 = [t for t in lines[:split_idx]]
    group2 = [t for t in lines[split_idx:]]

    # If one side is tiny, fallback to splitting by median
    if len(group1) < 2 or len(group2) < 2:
        med = float(np.median(cs))
        group1 = [t for t in lines if t[2] <= med]
        group2 = [t for t in lines if t[2] > med]

    def group_points(g):
        pts = []
        for (_, _, _, _, s) in g:
            x1, y1, x2, y2 = s
            pts.append((x1, y1))
            pts.append((x2, y2))
        return pts

    lineA = fit_line_through_points(group_points(group1))
    lineB = fit_line_through_points(group_points(group2))

    return lineA, lineB


def lane_pair_to_slot_pose(line1, line2, roi_shape, y_target_ratio=0.70, slot_len_ratio=0.35):
    """
    Convert two boundary lines into a slot pose (center_x, center_y, theta) 
    in ROI pixel coords.

    Args:
        line1, line2: (a, b, c, angle_deg) in ax+by+c=0 with unit-normal (a, b)
        roi_shape: (H, W) shape of ROI
        y_target_ratio: where to compute center (0.70 = 70% down ROI)
        slot_len_ratio: slot length as fraction of ROI height

    Returns:
        dict with keys: center, theta, width, length
    """
    H, W = roi_shape[:2]
    y = float(y_target_ratio * H)

    def x_at_y(line, y):
        a, b, c, _ = line
        if abs(a) < 1e-6:
            # nearly horizontal line in normal form; not expected for parking boundaries
            return None
        return -(b * y + c) / a

    x1 = x_at_y(line1, y)
    x2 = x_at_y(line2, y)
    if x1 is None or x2 is None:
        return None

    # ensure left/right ordering
    xl, xr = (x1, x2) if x1 < x2 else (x2, x1)

    center_x = 0.5 * (xl + xr)
    center_y = y

    # midline direction: direction angle is perpendicular to normal (a, b)
    def dir_angle_rad(line):
        a, b, _, _ = line
        # normal angle = atan2(b, a). Direction = normal + 90deg
        return np.arctan2(b, a) + np.pi / 2

    theta = float((dir_angle_rad(line1) + dir_angle_rad(line2)) * 0.5)

    # width (pixels): distance between two lines at y
    width = float(abs(xr - xl))

    # slot length in pixels: for now define as a fraction of ROI height
    length = float(slot_len_ratio * H)

    return {
        "center": (center_x, center_y),
        "theta": theta,   # radians
        "width": width,
        "length": length
    }


def slot_pose_to_corners(slot):
    """
    Build rectangle corners from (center, theta, width, length).
    Returns 4 corners in ROI pixel coords in order:
    [front-left, front-right, rear-right, rear-left]
    """
    cx, cy = slot["center"]
    th = slot["theta"]
    w = slot["width"]
    L = slot["length"]

    # local rectangle axes: forward along theta, left perpendicular
    fx, fy = np.cos(th), np.sin(th)
    lx, ly = -np.sin(th), np.cos(th)

    # half-dims
    hw = 0.5 * w
    hL = 0.5 * L

    # corners: front-left, front-right, rear-right, rear-left
    corners = []
    for sL, sW in [(+hL, +hw), (+hL, -hw), (-hL, -hw), (-hL, +hw)]:
        x = cx + sL * fx + sW * lx
        y = cy + sL * fy + sW * ly
        corners.append((float(x), float(y)))
    return corners
