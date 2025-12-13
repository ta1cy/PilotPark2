# hough_lane_detector.py

import cv2
import numpy as np


class HoughLaneDetector:
    def __init__(self, params=None):
        # Default parameters
        defaults = {
            # Hough
            "threshold": 10,     # very low, detect many segments
            "minLineLength": 10,
            "maxLineGap": 50,

            # initial angle filter (absolute angle in degrees)
            "angle_min": 20,    # # accept more angles
            "angle_max": 160,

            # ignore very short segments (in pixels, after ROI crop)
            "min_length": 20,   # almost no length filter

            # parallel pair constraints
            "angle_pair_th": 30.0,   # max angle diff between two lines (deg)
            "min_pair_gap": 10.0,   # min distance between two lane lines (pixels)
            "max_pair_gap": 400.0,  # max distance between two lane lines (pixels)
        }
        defaults.update(params or {})
        self.params = defaults

    # ---------- helpers ----------

    def _segments_to_params(self, segments):
        """
        Convert (x1,y1,x2,y2) segments to line params:
        angle, length, and normalized (a,b,c) for ax+by+c=0.
        """
        min_length = self.params["min_length"]
        params_list = []

        for (x1, y1, x2, y2) in segments:
            dx = x2 - x1
            dy = y2 - y1
            length = np.hypot(dx, dy)
            if length < min_length:
                continue

            angle = np.degrees(np.arctan2(dy, dx))  # -180..180

            # (a,b) is a unit normal to the line
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

    def _find_parallel_pairs(self, params_list):
        """
        Find pairs of nearly-parallel lines at a plausible lane gap.
        """
        angle_pair_th = self.params["angle_pair_th"]
        min_gap = self.params["min_pair_gap"]
        max_gap = self.params["max_pair_gap"]

        n = len(params_list)
        pairs = []
        used_idxs = set()

        for i in range(n):
            for j in range(i + 1, n):
                p1 = params_list[i]
                p2 = params_list[j]

                # angle difference (handle wrap-around)
                da = abs(p1["angle"] - p2["angle"])
                da = min(da, 180 - da)
                if da > angle_pair_th:
                    continue

                # distance between two parallel lines: |c1 - c2|
                dist = abs(p1["c"] - p2["c"])
                if dist < min_gap or dist > max_gap:
                    continue

                pairs.append((i, j, dist, da))
                used_idxs.add(i)
                used_idxs.add(j)

        # Optional debug:
        # print("num candidate lines:", n, "pairs found:", len(pairs))
        # for (i, j, dist, da) in pairs:
        #     print("pair", i, j, "gap:", dist, "angle diff:", da)

        return pairs, used_idxs

    # ---------- main API ----------

    def detect(self, edges):
        """
        Run HoughLinesP, basic angle filter, then keep ONLY lines
        that belong to a parallel pair with a fixed gap.
        """
        p = self.params
        threshold     = p['threshold']
        minLineLength = p['minLineLength']
        maxLineGap    = p['maxLineGap']
        angle_min     = p['angle_min']
        angle_max     = p['angle_max']

        line_img = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        # 1) Run Hough
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

        # 2) Basic angle filter -> candidate segments
        candidate_segments = []
        for l in raw_lines:
            x1, y1, x2, y2 = l[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            if angle_min <= abs(angle) <= angle_max:
                candidate_segments.append((x1, y1, x2, y2))

        if not candidate_segments:
            return line_img, []

        # 3) Convert segments to line params and find parallel pairs
        params_list = self._segments_to_params(candidate_segments)
        if not params_list:
            return line_img, []

        pairs, used_idxs = self._find_parallel_pairs(params_list)

        # 4) Draw only segments that are part of at least one good pair
        filtered_segments = []
        for idx, lp in enumerate(params_list):
            if idx in used_idxs:
                x1, y1, x2, y2 = lp["seg"]
                filtered_segments.append((x1, y1, x2, y2))
                cv2.line(line_img, (x1, y1), (x2, y2), (0, 255, 0), 3)

        return line_img, filtered_segments
