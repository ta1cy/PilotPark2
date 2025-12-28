# PilotPark2: Rear Camera Parking Slot Detection

A computer vision system for detecting parking slots from rear camera images using classical OpenCV techniques, designed to provide geometric pose information for autonomous parking planners.

## Overview

This project detects parking slot lanes and computes their geometric pose (center, orientation, dimensions) suitable for feeding into a parking path planner. The system uses LSD (Line Segment Detector) with angle-based filtering and robust line fitting to handle real-world noise and outliers.

## Features

### 1. Interactive Processing Pipeline
8-step visualization with real-time processing:
- **Load Image** - Load test image from samples folder
- **Grayscale** - Convert to single channel
- **Blur** - Gaussian blur for noise reduction
- **Binary** - Otsu's automatic thresholding
- **LSD Detection** - Line Segment Detector finds all line segments
- **Clustering** - Angle-based filtering separates left/right parking lines
- **Line Fitting** - Robust least squares fitting with outlier rejection
- **Pose Extraction** - Compute slot center, orientation, and dimensions

### 2. LSD Line Detection
- **Line Segment Detector (LSD)** - Direct line detection on binary images
- **No parameter tuning needed** - Works automatically on clean binary input
- **High precision** - Subpixel accuracy for line endpoints
- **Fast processing** - Efficient C++ implementation in OpenCV

### 3. Angle-Based Filtering
Separates parking lines from horizontal noise:
- **Left lines**: -85° to -20° (steep left-leaning)
- **Right lines**: 20° to 85° (steep right-leaning)
- **Ignored**: -20° to 20° (horizontal noise from ground markings)

### 4. Robust Line Fitting (`fit_line_to_segments`)
Three-layer outlier rejection:
- **Horizon filter** - Ignores segments in top 40% of image (y < 140)
- **Length filter** - Ignores tiny segments < 15 pixels
- **Segment weighting** - Longer segments (>30px) get more influence
- **Huber fitting** - `cv2.DIST_HUBER` resists remaining outliers better than L2

### 5. Parking Slot Pose Estimation (`slot_pose.py`)
Converts detected lane segments into actionable geometric data:
```python
{
    "center": (x, y),      # Slot center in ROI pixel coordinates
    "theta": θ,            # Orientation in radians
    "width": w,            # Pixels between boundaries
    "length": l            # Slot depth (configurable)
}
```

## Project Structure

```
PilotPark2/
├── rear_camera_view.py          # Main GUI application with 8-step pipeline
├── hough_lane_detector.py       # (Legacy) Hough-based detection
├── slot_pose.py                 # Geometric pose computation
├── lane_prior.py                # Lane prior creation utilities
├── auto_calibrate.py            # Auto parameter tuning utilities
├── samples/                     # Test images folder
├── result/                      # Screenshot results
└── README.md                    # This file
```

## Usage

```bash
python rear_camera_view.py
```

### Step-by-Step Processing
The GUI provides 8 buttons split across 2 rows for interactive pipeline visualization:

**Row 1: Preprocessing**
1. **Load Image** - Select test image from `samples/` folder
2. **Grayscale** - Convert to single channel (gray = 0.299R + 0.587G + 0.114B)
3. **Blur** - Gaussian blur (5x5 kernel) for noise reduction
4. **Binary** - Otsu's automatic thresholding (separates lines from background)
5. **Show Edges** - LSD line detection (displays all 50+ segments in light gray)

**Row 2: Line Analysis**
6. **Show Clustering** - Angle-based filtering (left=-85° to -20°, right=20° to 85°)
   - Draws on clean black background for clarity
   - Typical result: 1 left line + 3 right lines from 54 total segments
7. **Show Line Fitting** - Robust least squares fitting
   - Reuses clustering visualization as background
   - Draws infinite fitted lines through parking boundaries
   - Applies horizon/length/weighting filters to reject outliers
8. **Show Pose** - Extract parking slot geometry
   - Blue rectangle: parking slot boundary
   - Red dot: slot center
   - Yellow arrow: forward orientation

### Visual Pipeline
```
RGB Image → Gray → Blur → Binary → LSD Detection
                                        ↓
                                    54 segments
                                        ↓
                              Angle Filtering (-85° to -20° | 20° to 85°)
                                        ↓
                               4 valid parking lines
                                        ↓
                         Robust Fitting (horizon + length + Huber)
                                        ↓
                         2 infinite boundary lines
                                        ↓
                              Parking Slot Pose
```

## Output Visualization

- **Light gray lines** (Step 5) - All LSD detected segments before filtering
- **Colored segments on black** (Step 6) - Left (one color) and right (another color) groups after angle filtering
- **Infinite fitted lines** (Step 7) - Clean mathematical lines through parking boundaries
- **Blue rectangle** - Parking slot bounding box
- **Red dot** - Slot center point
- **Yellow arrow** - Forward orientation (theta angle)

## Key Design Decisions

### Why LSD Instead of Hough?
- **No parameter tuning** - LSD works automatically on clean binary images
- **Subpixel accuracy** - Better precision for line endpoints
- **Direct line detection** - No need to accumulate edge pixels
- **Fewer false positives** - Only detects actual line structures

### Why Angle-Based Filtering?
Initial clustering attempted to group by position (offset c), but horizontal noise segments interfered:
- **Problem**: Ground markings at top of image (~52 horizontal segments)
- **Solution**: Filter by angle first, then cluster by position
- **Result**: Cleanly separates left parking line from right parking lines

### Why Three-Layer Outlier Rejection?
1. **Horizon filter** - Top of image contains distant ground markings (not parking lines)
2. **Length filter** - Tiny segments (<15px) are usually noise from texture/shadows
3. **Segment weighting** - Longer segments are more reliable, should dominate the fit
4. **Huber fitting** - Robust estimator that down-weights remaining outliers automatically

This prevents the "diagonal line error" where outliers at the top of the image pull the fitted line away from the true parking boundary.

## Technical Details

### Image Processing Pipeline
```python
# 1. Load and crop to canonical size
img = cv2.imread("samples/test.jpg")
img = img[crop_y:crop_y+360, crop_x:crop_x+640]  # Center crop to 640x360

# 2. Preprocessing
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
_, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# 3. LSD detection
lsd = cv2.createLineSegmentDetector(0)
lines = lsd.detect(binary)[0]  # Returns [[x1, y1, x2, y2], ...]

# 4. Angle-based filtering
angle = math.atan2(y2 - y1, x2 - x1) * 180 / math.pi
if -85 <= angle <= -20:
    left_lines.append(segment)
elif 20 <= angle <= 85:
    right_lines.append(segment)

# 5. Robust line fitting
def fit_line_to_segments(segments):
    # Filter by horizon (y < 140) and length (< 15px)
    # Weight longer segments more heavily
    # Use cv2.fitLine with cv2.DIST_HUBER
    return (a, b, c, angle)  # General form: ax + by + c = 0

# 6. Pose extraction
slot_pose = compute_slot_from_lines(left_line, right_lines)
```

### Geometric Representation
- Lines in general form: `ax + by + c = 0` where `(a, b)` forms the normal vector
- Angle: `θ = atan2(vy, vx)` from line direction vector
- Distance between parallel lines: `|c₁ - c₂| / sqrt(a² + b²)`
- Clustering by offset `c` to separate left/right boundaries

### Outlier Rejection Strategy
The `fit_line_to_segments` function implements a three-stage filter:

```python
# Stage 1: Horizon filter
HORIZON_Y = 140  # Ignore top 40% of 360px image
if y1 < HORIZON_Y and y2 < HORIZON_Y:
    continue

# Stage 2: Length filter
MIN_LENGTH = 15
length = np.hypot(x2 - x1, y2 - y1)
if length < MIN_LENGTH:
    continue

# Stage 3: Segment weighting
points.append([x1, y1])
points.append([x2, y2])
if length > 30:  # Long segments get 3x weight
    points.append([x1, y1])
    points.append([x2, y2])
    points.append([(x1+x2)/2, (y1+y2)/2])

# Stage 4: Robust fitting
vx, vy, x0, y0 = cv2.fitLine(points, cv2.DIST_HUBER, 0, 0.01, 0.01)
```

This prevents the "outlier interference" problem where small noise segments at the top of the image pull the fitted line away from the actual parking boundary.

## Integration with Parking Planner

The system outputs pose data ready for path planning algorithms:

```python
# Perception module output
slot_pose = detect_slot_pose(frame)
# → Returns: {
#     "center": (x, y),
#     "theta": θ,
#     "width": w,
#     "length": l
# }

# Feed to planner
target_pose = (slot_pose["center"][0], 
               slot_pose["center"][1], 
               slot_pose["theta"])
target_box = (slot_pose["width"], slot_pose["length"])

# Planner computes trajectory from current car pose to target_pose
```

## Development Process

### Evolution of the Pipeline
1. **Initial approach**: Hough transform with parallel pair filtering
   - Required extensive parameter tuning
   - False positives from ground markings
   
2. **Switch to LSD**: Line Segment Detector
   - Automatic detection on binary images
   - No parameter tuning needed
   - Higher precision

3. **Clustering challenges**: Position-based clustering failed
   - 52 horizontal noise segments grouped with left parking line
   - Solution: Filter by angle BEFORE clustering by position

4. **Outlier interference**: Tiny noise at top of image
   - Small yellow dashes pulled fitted line away from true boundary
   - Solution: Three-layer filter (horizon + length + weighting) + Huber fitting

5. **Final pipeline**: Robust and automatic
   - Angle filtering: -85° to -20° (left) | 20° to 85° (right)
   - Outlier rejection: horizon + length + segment weighting
   - Robust fitting: cv2.DIST_HUBER for outlier resistance

### Key Lessons Learned
- **Geometric filtering first**: Filter by angle before clustering by position
- **Multiple outlier defenses**: Single-layer filtering is insufficient for real-world noise
- **Robust estimators**: Huber distance outperforms L2 for line fitting with outliers
- **Visual continuity**: Reuse visualization images between steps for better UX

## Next Steps

- **Bird's Eye View (BEV) transformation** - Convert from image coordinates to metric ground plane
- **Calibration utilities** - Camera intrinsic/extrinsic calibration tools
- **Multiple slot detection** - Handle multiple parking slots in one image
- **Temporal filtering** - Smooth detections across video frames
- **Simulation integration** - Test with CARLA/Gazebo simulator
- **Sensor fusion** - Combine camera with ultrasonic/radar distance sensors
- **Path planning** - Implement Reeds-Shepp curves or MPC for trajectory generation
- **Real vehicle testing** - Deploy on Honda Pilot with aftermarket sensors

---

**A classical computer vision approach that successfully bridges perception and planning for autonomous parking, using robust outlier rejection and angle-based filtering to handle real-world noise.**
