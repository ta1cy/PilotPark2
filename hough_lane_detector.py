# hough_lane_detector.py
import cv2
import numpy as np


class HoughLaneDetector:
    """
    Hough lane detector with "pair filtering":
      - detect segments with HoughLinesP
      - filter by angle range
      - keep only segments that belong to at least one parallel pair
        whose distance lies in [min_pair_gap, max_pair_gap]

    New: auto scaling of pixel thresholds based on ROI size (edges.shape).
    """

    def __init__(self, params=None):
        defaults = {
            # ---- Hough ----
            "threshold": 10,
            "minLineLength": 10,   # if None and auto_scale=True -> computed from ROI height
            "maxLineGap": 10,

            # ---- angle filter on segments (absolute degrees) ----
            "angle_min": 15,
            "angle_max": 100,

            # ---- post filter ----
            "min_length": 27,      # if None and auto_scale=True -> computed from ROI height

            # ---- pair constraints ----
            "angle_pair_th": 1.0,
            "min_pair_gap": 19,    # if None and auto_scale=True -> computed from ROI width
            "max_pair_gap": 21,    # if None and auto_scale=True -> computed from ROI width

            # ---- auto scaling controls ----
            "auto_scale": True,

            # ratios used when auto_scale is True and the corresponding value is None
            # (tune these once and they should generalize across crops/resolutions)
            "minLineLength_ratio_h": 0.40,  # 40% of ROI height
            "min_length_ratio_h":    0.30,  # 30% of ROI height
            "min_pair_gap_ratio_w":  0.10,  # 10% of ROI width
            "max_pair_gap_ratio_w":  0.45,  # 45% of ROI width

            # safety clamps (pixels)
            "minLineLength_min_px": 40,
            "min_length_min_px":    30,
            "min_pair_gap_min_px":  10,
            "max_pair_gap_min_px":  50,
        }
        defaults.update(params or {})
        self.params = defaults

    # ---------- helpers ----------

    def _resolve_scaled_params(self, edges_shape):
        """
        Compute effective parameters, scaling pixel thresholds with ROI size
        when auto_scale=True and the parameter is None.
        """
        p = self.params
        H, W = edges_shape[:2]

        auto = bool(p.get("auto_scale", True))

        # Hough minLineLength
        if auto and p.get("minLineLength") is None:
            mll = int(p["minLineLength_ratio_h"] * H)
            mll = max(mll, int(p["minLineLength_min_px"]))
        else:
            mll = int(p["minLineLength"]) if p.get("minLineLength") is not None else 40

        # post-filter min_length
        if auto and p.get("min_length") is None:
            ml = int(p["min_length_ratio_h"] * H)
            ml = max(ml, int(p["min_length_min_px"]))
        else:
            ml = int(p["min_length"]) if p.get("min_length") is not None else 30

        # pair gaps
        if auto and p.get("min_pair_gap") is None:
            ming = int(p["min_pair_gap_ratio_w"] * W)
            ming = max(ming, int(p["min_pair_gap_min_px"]))
        else:
            ming = float(p["min_pair_gap"]) if p.get("min_pair_gap") is not None else 10.0

        if auto and p.get("max_pair_gap") is None:
            maxg = int(p["max_pair_gap_ratio_w"] * W)
            maxg = max(maxg, int(p["max_pair_gap_min_px"]))
        else:
            maxg = float(p["max_pair_gap"]) if p.get("max_pair_gap") is not None else float(W)

        # ensure valid ordering
        if maxg <= ming:
            maxg = ming + 1.0

        resolved = {
            "threshold": int(p["threshold"]),
            "minLineLength": int(mll),
            "maxLineGap": int(p["maxLineGap"]),
            "angle_min": float(p["angle_min"]),
            "angle_max": float(p["angle_max"]),
            "min_length": float(ml),
            "angle_pair_th": float(p["angle_pair_th"]),
            "min_pair_gap": float(ming),
            "max_pair_gap": float(maxg),
        }
        return resolved

    def _segments_to_params(self, segments, min_length):
        """
        Convert (x1,y1,x2,y2) segments to line params:
        angle, length, and normalized (a,b,c) for ax+by+c=0.
        """
        params_list = []
        for (x1, y1, x2, y2) in segments:
            dx = x2 - x1
            dy = y2 - y1
            length = np.hypot(dx, dy)
            if length < min_length:
                continue

            angle = np.degrees(np.arctan2(dy, dx))  # -180..180

            # unit normal (a,b) to line
            a = dy
            b = -dx
            norm = np.hypot(a, b)
            if norm == 0:
                continue
            a /= norm
            b /= norm
            c = -(a * x1 + b * y1)

            params_list.append({
                "seg": (x1, y1, x2, y2),
                "angle": angle,
                "length": length,
                "a": a, "b": b, "c": c
            })
        return params_list

    def _find_parallel_pairs(self, params_list, angle_pair_th, min_gap, max_gap):
        """
        Find pairs that are nearly parallel and separated by a distance in [min_gap, max_gap].
        Returns (pairs, used_idxs).
        """
        n = len(params_list)
        pairs = []
        used = set()

        for i in range(n):
            for j in range(i + 1, n):
                p1 = params_list[i]
                p2 = params_list[j]

                # angle diff, wrap-aware
                da = abs(p1["angle"] - p2["angle"])
                da = min(da, 180 - da)
                if da > angle_pair_th:
                    continue

                # since (a,b) are unit normals, parallel-line distance is |c1 - c2|
                dist = abs(p1["c"] - p2["c"])
                if dist < min_gap or dist > max_gap:
                    continue

                pairs.append((i, j, dist, da))
                used.add(i)
                used.add(j)

        return pairs, used

    # ---------- main API ----------

    def detect(self, edges):
        """
        Input: edges (single-channel Canny result)
        Output: (line_img, filtered_segments)
        """
        eff = self._resolve_scaled_params(edges.shape)

        threshold     = eff["threshold"]
        minLineLength = eff["minLineLength"]
        maxLineGap    = eff["maxLineGap"]

        angle_min     = eff["angle_min"]
        angle_max     = eff["angle_max"]

        min_length    = eff["min_length"]
        angle_pair_th = eff["angle_pair_th"]
        min_pair_gap  = eff["min_pair_gap"]
        max_pair_gap  = eff["max_pair_gap"]

        line_img = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        raw_lines = cv2.HoughLinesP(
            edges,
            1,
            np.pi / 180,
            threshold=threshold,
            minLineLength=minLineLength,
            maxLineGap=maxLineGap
        )
        if raw_lines is None:
            return line_img, []

        # 1) angle filter -> candidate segments
        candidate = []
        for l in raw_lines:
            x1, y1, x2, y2 = l[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            if angle_min <= abs(angle) <= angle_max:
                candidate.append((x1, y1, x2, y2))

        if not candidate:
            return line_img, []

        # 2) convert to infinite-line params & pair
        params_list = self._segments_to_params(candidate, min_length=min_length)
        if not params_list:
            return line_img, []

        _, used_idxs = self._find_parallel_pairs(
            params_list,
            angle_pair_th=angle_pair_th,
            min_gap=min_pair_gap,
            max_gap=max_pair_gap
        )

        # 3) draw only segments belonging to a valid pair
        filtered = []
        for idx, lp in enumerate(params_list):
            if idx in used_idxs:
                x1, y1, x2, y2 = lp["seg"]
                filtered.append((x1, y1, x2, y2))
                cv2.line(line_img, (x1, y1), (x2, y2), (0, 255, 0), 3)

        return line_img, filtered
