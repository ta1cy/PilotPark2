# PilotPark2: Rear Camera Parking Slot Detection

A computer vision system for detecting parking slots from rear camera images using classical OpenCV techniques, designed to provide geometric pose information for autonomous parking planners.

## Overview

This project detects parking slot lanes and computes their geometric pose (center, orientation, dimensions) suitable for feeding into a parking path planner. The system uses parallel line pair detection with intelligent filtering and geometric fitting.

## Features

### 1. Interactive Processing Pipeline
8-step visualization with real-time parameter tuning:
- **ROI Cropping** - Fixed region of interest (bottom 60% of frame)
- **Grayscale Conversion** - Simplify color information
- **Gaussian Blur** - Noise reduction
- **Canny Edge Detection** - Edge extraction
- **Hough Line Detection** - Segment detection with parallel pair filtering
- **Segment Clustering** - Group segments into left/right boundaries
- **Line Fitting** - Fit clean lines to reduce jitter
- **Pose Extraction** - Compute slot center, orientation, and dimensions

### 2. Smart Lane Detection (`hough_lane_detector.py`)
- **HoughLinesP** segment detection
- **Angle filtering** - Removes horizontal noise (keeps 15-100°)
- **Length filtering** - Drops short segments
- **Parallel pair detection** - Only keeps segments forming valid lane pairs
- **Auto-scaling** - Parameters adapt to ROI size for different resolutions

### 3. Parking Slot Pose Estimation (`slot_pose.py`)
Converts detected lane segments into actionable geometric data:
```python
{
    "center": (x, y),      # Slot center in ROI pixel coordinates
    "theta": θ,            # Orientation in radians
    "width": w,            # Pixels between boundaries
    "length": l            # Slot depth (configurable)
}
```

### 4. Real-Time Parameter Tuning
9 interactive sliders for live adjustment:
- Hough parameters: `threshold`, `minLineLength`, `maxLineGap`
- Angle constraints: `angle_min`, `angle_max`
- Post-filtering: `min_length`, `angle_pair_th`
- Pair spacing: `min_pair_gap`, `max_pair_gap`

## Project Structure

```
PilotPark2/
├── rear_camera_view.py          # Main GUI application
├── hough_lane_detector.py       # Parallel pair lane detection
├── slot_pose.py                 # Geometric pose computation
├── test_parking_1.jpg           # Test image
├── result/                      # Screenshot results
└── parking-assistant-proposal.md # Project roadmap
```

## Usage

```bash
python rear_camera_view.py
```

### Step-by-Step Mode
Click buttons sequentially to see each processing stage:
1. Process image through edge detection
2. Detect lane segments with Hough transform
3. Cluster segments into boundaries (green/cyan visualization)
4. Fit clean infinite lines
5. Extract parking slot pose (blue box + red center + yellow orientation arrow)

### Quick Mode
Click **"Process All Steps"** to run the complete pipeline at once.

### Parameter Tuning
1. Adjust sliders at the bottom of the GUI
2. Click **"Update Parameters"**
3. Re-run **"Hough Lane Detection"** to see changes

## Output Visualization

- **Green lines** - Detected lane segments (after filtering)
- **Green/Cyan segments** - Clustered boundary groups
- **Green/Cyan infinite lines** - Fitted boundary lines
- **Blue rectangle** - Parking slot box
- **Red dot** - Slot center point
- **Yellow arrow** - Forward orientation (theta)

## Parameter Tuning Strategy

1. **Start with generous thresholds** to see plenty of detections
2. **Increase `min_length`** (e.g. 27 → 60) to remove short noisy segments
3. **Tighten `angle_pair_th`** (e.g. 30 → 15 → 1) to keep only parallel pairs
4. **Narrow `[min_pair_gap, max_pair_gap]`** to match actual lane width
   - Measure approximate lane spacing in pixels (e.g. ~80–150)
   - Try `min_pair_gap = 60`, `max_pair_gap = 180`
5. **If too many segments remain:**
   - Increase `threshold` (e.g. 20, 30...)
   - Or increase `minLineLength`

**Tune in this order for best results.**

## Integration with Parking Planner

The system outputs pose data ready for path planning algorithms:

```python
# Perception module output
slot_pose = detect_slot_pose(frame)
# → Returns: (center_x, center_y, theta, width, length)

# Feed to planner
target_pose = (slot_pose["center"][0], 
               slot_pose["center"][1], 
               slot_pose["theta"])
target_box = (slot_pose["width"], slot_pose["length"])

# Planner computes trajectory from current car pose to target_pose
```

## Next Steps

- **Bird's Eye View (BEV) transformation** - Convert from image coordinates to metric ground plane
- **Simulation integration** - Test with CARLA/Gazebo simulator
- **Sensor fusion** - Combine camera with ultrasonic/radar distance sensors
- **Path planning** - Implement Reeds-Shepp curves or MPC for trajectory generation
- **Real vehicle testing** - Deploy on Honda Pilot with aftermarket sensors

## Technical Details

### Geometric Approach
- Lines represented in normal form: `ax + by + c = 0` where `(a,b)` is unit normal
- Distance between parallel lines: `|c₁ - c₂|` (when normals are consistent)
- Clustering by offset `c` to separate left/right boundaries
- Line fitting via `cv2.fitLine` with L2 distance minimization

### Auto-Scaling Design
Parameters scale with ROI dimensions using configurable ratios:
- `minLineLength` = 40% of ROI height
- `min_length` = 30% of ROI height  
- `min_pair_gap` = 10% of ROI width
- `max_pair_gap` = 45% of ROI width

This makes the detector robust to different camera angles and resolutions.

---

**A classical computer vision approach that bridges perception and planning for autonomous parking.**
