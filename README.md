# Parameter Tuning Strategy

1. **Start with generous thresholds** to see plenty of detections.
2. **Increase `min_length`** (e.g. 40 → 60) to remove short noisy segments.
3. **Tighten `angle_pair_th`** (e.g. 30 → 15 → 8) to keep only nicely parallel pairs.
4. **Narrow `[min_pair_gap, max_pair_gap]`**
    - Measure approximate lane spacing in pixels (e.g. ~80–150)
    - Try `min_pair_gap = 60`, `max_pair_gap = 180`.
5. **If there are still too many segments:**
    - Increase `threshold` (e.g. 20, 30…)
    - Or increase `minLineLength`.

**Tune in this order for best results.**
# Rear Camera Parking Slot Detection Playground

## Problem Statement

The goal is to detect parking slot lanes from a rear camera image using classical computer vision techniques (OpenCV). A parking lane is defined as two long, roughly parallel lines whose distance is within a fixed pixel range. Any line that doesn’t belong to such a pair should be dropped.

## Solution Approach

1. **Image Processing Pipeline**
    - Crop ROI (manual selection)
    - Grayscale conversion
    - Gaussian blur
    - Canny edge detection
    - Hough Line detection (with runtime-tunable parameters)

2. **Parking Lane Filtering**
    - Use HoughLinesP to detect all line segments.
    - For each pair of detected lines:
        - Check if both lines are long enough.
        - Calculate the angle of each line and ensure they are roughly parallel (angle difference within a threshold).
        - Compute the minimum distance between the two lines; keep pairs whose distance is within a fixed pixel range (e.g., 100-300 pixels).
    - Drop any line that does not belong to such a pair.

## Experience Record

- The pipeline is interactive, with a Tkinter GUI for stepwise processing and parameter tuning.
- Manual ROI cropping is supported for flexible region selection.
- Hough line parameters can be tuned live for best results.
- Filtering logic for parking lanes is planned to be added, leveraging the above conceptual approach.

## Next Steps

- Implement the parking lane filtering logic in the codebase.
- Visualize only valid parking lane pairs in the processed image.
- Further tune parameters and add pose estimation for detected slots.

---

**This README documents the problem, solution concept, and experience for future reference and improvement.**
