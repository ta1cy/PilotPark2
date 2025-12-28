"""
rear_camera_view.py

Interactive playground for rear camera image processing:
- Manual ROI crop
- Grayscale
- Blur
- Canny edges
- Hough lane detection with parallel-pair filtering (via HoughLaneDetector)
"""

from xml.dom import WrongDocumentErr
import cv2
import numpy as np
from tkinter import Tk, Button, Label, Checkbutton, IntVar, Frame, filedialog
from PIL import Image, ImageTk
import tkinter as tk

from hough_lane_detector import HoughLaneDetector  # separate file
from slot_pose import (
    cluster_into_two_boundaries,
    lane_pair_to_slot_pose,
    slot_pose_to_corners
)
from auto_calibrate import (
    auto_canny,
    auto_hough_threshold,
    auto_tune
)
from lane_prior import create_lane_prior
import math


def cv2_to_tk(img):
    """Convert OpenCV image (BGR or GRAY) to a Tkinter PhotoImage."""
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img)
    pil_img = pil_img.resize((400, 300))
    return ImageTk.PhotoImage(pil_img)


def fit_line_to_segments(segments):
    """
    Takes a list of segments and uses Least Squares to find the best fit line.
    Returns: (a, b, c, angle) for ax + by + c = 0
    """
    if not segments:
        return None

    # Collect all start and end points of the segments
    points = []
    for s in segments:
        x1, y1, x2, y2 = s
        points.append([x1, y1])
        points.append([x2, y2])
    
    points = np.array(points, dtype=np.float32)
    
    # Fit a line using OpenCV's fitLine (Distance based, robust to noise)
    # DIST_L2 is standard Least Squares
    vx, vy, x0, y0 = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01)
    
    # Convert vector (vx, vy) and point (x0, y0) to General Form: ax + by + c = 0
    # Slope m = vy / vx. 
    # General form is -vy*x + vx*y + (vy*x0 - vx*y0) = 0
    
    a = -vy[0]
    b = vx[0]
    c = vy[0] * x0[0] - vx[0] * y0[0]
    
    # Calculate angle for reference
    angle = math.degrees(math.atan2(vy[0], vx[0]))
    
    return (a, b, c, angle)


# ----------------------------------------------------------
# Image processor: crop → gray → blur → edges
# ----------------------------------------------------------

class ImageProcessor:
    # Canonical size for standardized input
    CANONICAL_WIDTH = 640
    CANONICAL_HEIGHT = 360
    
    def __init__(self, img):
        # Step 0: Standardize input - center crop to canonical size
        H, W = img.shape[:2]
        
        # Validate minimum size
        if H < self.CANONICAL_HEIGHT or W < self.CANONICAL_WIDTH:
            raise ValueError(f"Invalid image size {W}x{H}. Minimum required: {self.CANONICAL_WIDTH}x{self.CANONICAL_HEIGHT}")
        
        # Center crop to canonical size
        y_center = H // 2
        x_center = W // 2
        y0 = y_center - self.CANONICAL_HEIGHT // 2
        x0 = x_center - self.CANONICAL_WIDTH // 2
        y1 = y0 + self.CANONICAL_HEIGHT
        x1 = x0 + self.CANONICAL_WIDTH
        
        self.img = img[y0:y1, x0:x1].copy()
        print(f"Center cropped from {W}x{H} to {self.CANONICAL_WIDTH}x{self.CANONICAL_HEIGHT}")
        
        self.h, self.w = self.img.shape[:2]
        self.roi = None
        self.gray = None
        self.blur = None  # Gaussian blurred to remove noise
        self.binary = None  # Binarized image (white lanes only)
        self.edges = None
        self.lsd_lines = None  # Store LSD detected lines
        self.lane_mask = None  # DL lane prior mask

        # Initialize hough_params from detector defaults so they stay in sync
        self.hough_params = HoughLaneDetector().params.copy()
        
        # Step 1: Create LSD-based lane prior for rear camera
        print("\n=== Step 1: LSD Line Detection ===")
        self.lane_mask = create_lane_prior(self.img, use_lsd=True)
        
        # Show mask statistics
        mask_pixels = np.count_nonzero(self.lane_mask)
        mask_percent = 100 * mask_pixels / (self.lane_mask.shape[0] * self.lane_mask.shape[1])
        print(f"  Lane mask coverage: {mask_pixels} pixels ({mask_percent:.1f}%)")

    def step_crop(self):
        """
        ROI is now the full standardized image (no additional cropping).
        """
        self.roi = self.img
        return self.roi

    def step_gray(self):
        if self.roi is None:
            self.step_crop()
        self.gray = cv2.cvtColor(self.roi, cv2.COLOR_BGR2GRAY)
        return self.gray

    def step_blur(self):
        """Apply Gaussian blur to remove white noise dots."""
        if self.gray is None:
            self.step_gray()
        self.blur = cv2.GaussianBlur(self.gray, (5, 5), 0)
        return self.blur

    def step_binary(self):
        """Binarize to extract white/bright lanes only."""
        if self.blur is None:
            self.step_blur()
        # Use Otsu's thresholding to automatically find optimal threshold
        # This will keep white/bright pixels (lanes) and set dark pixels to black
        _, self.binary = cv2.threshold(self.blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return self.binary

    def step_edges(self, use_lsd=True):
        if self.binary is None:
            self.step_binary()
        
        if use_lsd:
            # Use LSD (Line Segment Detector) - direct line detection
            lsd = cv2.createLineSegmentDetector(0)
            lines = lsd.detect(self.binary)[0]
            
            # Store lines for later use
            self.lsd_lines = lines
            
            # Create edge image from detected lines with same color
            self.edges = np.zeros((self.gray.shape[0], self.gray.shape[1], 3), dtype=np.uint8)
            if lines is not None:
                for i, line in enumerate(lines):
                    x1, y1, x2, y2 = map(int, line[0])
                    # Draw all lines in light gray
                    cv2.line(self.edges, (x1, y1), (x2, y2), (192, 192, 192), 2)
                print(f"LSD detected {len(lines)} line segments")
            else:
                print("LSD: No line segments detected")
        else:
            # Fallback to Canny (original behavior)
            if self.binary is None:
                self.step_binary()
            self.edges, self.canny_thresholds = auto_canny(self.binary)
        
        return self.edges

    def process_all(self):
        self.step_crop()
        self.step_gray()
        self.step_blur()
        self.step_binary()
        self.step_edges()


# ----------------------------------------------------------
# Tkinter App
# ----------------------------------------------------------

class App:
    def __init__(self, root, img=None):
        self.root = root
        self.root.title("Rear Camera View Playground")
        self.processor = ImageProcessor(img) if img is not None else None
        self.current_img = img

        # Frames
        self.left_frame = Frame(self.root)
        self.left_frame.grid(row=0, column=0, padx=10, pady=10)
        self.right_frame = Frame(self.root)
        self.right_frame.grid(row=0, column=1, padx=10, pady=10)
        self.control_frame = Frame(self.root)
        self.control_frame.grid(row=1, column=0, columnspan=2, pady=10)

        # Original image
        orig_title = f"Original Image - {img.shape[1]}x{img.shape[0]}" if img is not None else "Original Image - 640x360"
        self.orig_label = Label(self.left_frame, text=orig_title, font=("Arial", 10, "bold"))
        self.orig_label.pack()
        self.orig_panel = Label(self.left_frame)
        self.orig_panel.pack()
        
        # Load initial image if provided, otherwise show black window
        if img is not None:
            self.orig_img = cv2_to_tk(img)
            self.orig_panel.config(image=self.orig_img)
            self.orig_panel.image = self.orig_img
        else:
            # Create black 640x360 placeholder
            black_img = np.zeros((360, 640, 3), dtype=np.uint8)
            self.orig_img = cv2_to_tk(black_img)
            self.orig_panel.config(image=self.orig_img)
            self.orig_panel.image = self.orig_img

        # Processed image
        proc_title = f"Processed Image - {img.shape[1]}x{img.shape[0]}" if img is not None else "Processed Image - 640x360"
        self.proc_label = Label(self.right_frame, text=proc_title, font=("Arial", 10, "bold"))
        self.proc_label.pack()
        self.proc_panel = Label(self.right_frame)
        self.proc_panel.pack()
        
        # Show black placeholder in processed panel too
        if img is None:
            black_img = np.zeros((360, 640, 3), dtype=np.uint8)
            self.proc_img = cv2_to_tk(black_img)
            self.proc_panel.config(image=self.proc_img)
            self.proc_panel.image = self.proc_img

        # State for slot detection steps
        self.segments = None
        self.lineA = None
        self.lineB = None
        self.slot = None
        
        # Auto-tuning state
        self.use_auto_tune = IntVar(value=1)  # Enable by default

        # Checklist buttons
        self.steps_row1 = [
            ("Load Image", self.load_image),
            ("Convert to Grayscale", self.show_gray),
            ("Gaussian Blur", self.show_blur),
            ("Binarize (White Lanes)", self.show_binary),
            ("LSD Line Detection", self.show_edges),
        ]
        self.steps_row2 = [
            ("Segment Clustering", self.show_clustering),
            ("Line Fitting", self.show_line_fitting),
            ("Pose Extraction", self.show_pose_extraction),
        ]
        
        self.vars = [IntVar() for _ in range(len(self.steps_row1) + len(self.steps_row2))]
        
        # First row of buttons
        for i, (name, func) in enumerate(self.steps_row1):
            btn = Button(self.control_frame, text=name, command=func)
            btn.grid(row=0, column=i, padx=5)
            chk = Checkbutton(self.control_frame, text="Done", variable=self.vars[i], state="disabled")
            chk.grid(row=1, column=i)
        
        # Second row of buttons (slot detection steps)
        for i, (name, func) in enumerate(self.steps_row2):
            var_idx = len(self.steps_row1) + i
            btn = Button(self.control_frame, text=name, command=func)
            btn.grid(row=2, column=i, padx=5)
            chk = Checkbutton(self.control_frame, text="Done", variable=self.vars[var_idx], state="disabled")
            chk.grid(row=3, column=i)

        
        # Auto-tune checkbox (second row, rightmost)
        self.auto_chk = Checkbutton(self.control_frame, text="Auto-Tune", 
                                     variable=self.use_auto_tune)
        self.auto_chk.grid(row=3, column=len(self.steps_row1)-1, sticky="e")
        # Button to process all (far right of second row)
        self.all_btn = Button(self.control_frame, text="Process All Steps", command=self.process_all)
        self.all_btn.grid(row=2, column=len(self.steps_row1)-1, padx=5, sticky="e")

        # Parameter sliders
        # self.add_param_controls()

    # ---- step display methods ----

    def load_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Parking Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp"),
                ("All files", "*.*")
            ],
            initialdir="samples"
        )
        
        if not file_path:
            return
        
        img = cv2.imread(file_path)
        if img is None:
            print(f"Error: Could not load image at {file_path}")
            return
        
        # Update current image and processor
        self.current_img = img
        self.processor = ImageProcessor(img)
        
        # Reset all checkboxes
        for v in self.vars:
            v.set(0)
        
        # Reset state
        self.segments = None
        self.lineA = None
        self.lineB = None
        self.slot = None
        
        # Update window titles with resolution
        orig_h, orig_w = img.shape[:2]
        proc_h, proc_w = self.processor.img.shape[:2]
        self.orig_label.config(text=f"Original Image - {orig_w}x{orig_h}")
        self.proc_label.config(text=f"Processed Image - {proc_w}x{proc_h}")
        
        # Display original image
        self.orig_img = cv2_to_tk(img)
        self.orig_panel.config(image=self.orig_img)
        self.orig_panel.image = self.orig_img
        
        # Show ROI crop in processed panel
        roi = self.processor.step_crop()
        self.proc_img = cv2_to_tk(roi)
        self.proc_panel.config(image=self.proc_img)
        self.proc_panel.image = self.proc_img
        
        print(f"Loaded image: {file_path}")
        print(f"Image size: {img.shape[1]}x{img.shape[0]}")
        self.vars[0].set(1)

    def show_lane_mask(self):
        if self.processor is None:
            print("Please load an image first!")
            return
        
        if self.processor.lane_mask is None:
            print("Lane mask not available!")
            return
        
        # Visualize lane mask (colorize for better visibility)
        lane_mask_color = cv2.cvtColor(self.processor.lane_mask, cv2.COLOR_GRAY2BGR)
        lane_mask_color[self.processor.lane_mask > 0] = [0, 255, 0]  # Green lanes
        
        # Blend with original image
        blended = cv2.addWeighted(self.processor.img, 0.6, lane_mask_color, 0.4, 0)
        
        self.proc_img = cv2_to_tk(blended)
        self.proc_panel.config(image=self.proc_img)
        self.proc_panel.image = self.proc_img
        self.vars[2].set(1)
        
        print("Displayed LSD lane prior mask")

    def show_gray(self):
        if self.processor is None:
            print("Please load an image first!")
            return
        gray = self.processor.step_gray()
        self.proc_img = cv2_to_tk(gray)
        self.proc_panel.config(image=self.proc_img)
        self.proc_panel.image = self.proc_img
        self.vars[1].set(1)

    def show_blur(self):
        if self.processor is None:
            print("Please load an image first!")
            return
        blur = self.processor.step_blur()
        self.proc_img = cv2_to_tk(blur)
        self.proc_panel.config(image=self.proc_img)
        self.proc_panel.image = self.proc_img
        self.vars[2].set(1)

    def show_binary(self):
        if self.processor is None:
            print("Please load an image first!")
            return
        binary = self.processor.step_binary()
        self.proc_img = cv2_to_tk(binary)
        self.proc_panel.config(image=self.proc_img)
        self.proc_panel.image = self.proc_img
        self.vars[3].set(1)

    def show_edges(self):
        if self.processor is None:
            print("Please load an image first!")
            return
        edges = self.processor.step_edges(use_lsd=True)
        
        # Store LSD segments for later use
        if self.processor.lsd_lines is not None:
            # Convert LSD lines to segments format
            self.segments = []
            for line in self.processor.lsd_lines:
                x1, y1, x2, y2 = line[0]
                self.segments.append((x1, y1, x2, y2))
        
        self.proc_img = cv2_to_tk(edges)
        self.proc_panel.config(image=self.proc_img)
        self.proc_panel.image = self.proc_img
        self.vars[4].set(1)

    def show_clustering(self):
        if self.segments is None or len(self.segments) < 2:
            print("Run LSD Line Detection first!")
            return

        # Use already computed edges
        if self.processor.edges is None:
            return
            
        # Create clean black image for clustering visualization
        H, W = self.processor.edges.shape[:2]
        line_img = np.zeros((H, W, 3), dtype=np.uint8)

        import math

        # --- NEW LOGIC: Cluster by Angle (Slope) ---
        group1 = [] # Left Lines (Green)
        group2 = [] # Right Lines (Cyan)
        
        # Temporary lists to help fit the infinite lines later (lineA, lineB)
        group1_segments = []
        group2_segments = []

        for s in self.segments:
            x1, y1, x2, y2 = s
            
            # 1. Calculate Angle in Degrees
            # atan2 returns -180 to 180
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
            
            # 2. Filter Logic
            # Note: Image coordinates (0,0 at top-left) means:
            # Right Line (\) is usually +45 to +85 degrees
            # Left Line (/) is usually -45 to -85 degrees
            
            # CHECK FOR RIGHT LINE (Cyan) - Slanting \
            if 20 < angle < 85:
                group2.append(s)
                group2_segments.append(s)
                
            # CHECK FOR LEFT LINE (Green) - Slanting /
            # Also handle the wrap-around case if line points up vs down
            elif -85 < angle < -20:
                group1.append(s)
                group1_segments.append(s)
                
            # Horizontal lines (-20 to 20) are IGNORED (Noise)
            # Vertical lines (>85) are IGNORED (Noise)

        # --- DRAWING ---
        for s in group1:
            x1, y1, x2, y2 = s
            cv2.line(line_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)  # Green (Left)

        for s in group2:
            x1, y1, x2, y2 = s
            cv2.line(line_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)  # Cyan (Right)

        print(f"Clustered by Angle: {len(group1)} (Left/Green) and {len(group2)} (Right/Cyan)")

        # --- THE MISSING CALCULATION STEP ---
        # We must turn the LIST of segments into a single MATHEMATICAL line (a,b,c)
        # and store it in self.lineA / self.lineB so the next function can use it.
        
        if len(group1) > 0:
            self.lineA = fit_line_to_segments(group1)  # Updates the Left Line
        else:
            self.lineA = None  # No left line found
            
        if len(group2) > 0:
            self.lineB = fit_line_to_segments(group2)  # Updates the Right Line
        else:
            self.lineB = None  # No right line found

        # Store clustering image for line fitting step
        self.clustering_img = line_img.copy()

        self.proc_img = cv2_to_tk(line_img)
        self.proc_panel.config(image=self.proc_img)
        self.proc_panel.image = self.proc_img
        self.vars[5].set(1)

    def show_line_fitting(self):
        if self.lineA is None or self.lineB is None:
            print("Run Segment Clustering first!")
            return

        # Use clustering visualization as background
        if not hasattr(self, 'clustering_img') or self.clustering_img is None:
            print("Run Segment Clustering first!")
            return
            
        # Copy clustering image to draw fitted lines on top
        line_img = self.clustering_img.copy()

        # Draw the fitted infinite lines across the image
        H, W = line_img.shape[:2]

        def draw_fitted_line(img, line, color):
            a, b, c, _ = line
            # Draw line by finding intersections with image boundaries
            # ax + by + c = 0
            points = []
            
            # Top edge (y=0)
            if abs(a) > 1e-6:
                x = -(b * 0 + c) / a
                if 0 <= x <= W:
                    points.append((int(x), 0))
            
            # Bottom edge (y=H)
            if abs(a) > 1e-6:
                x = -(b * H + c) / a
                if 0 <= x <= W:
                    points.append((int(x), H))
            
            # Left edge (x=0)
            if abs(b) > 1e-6:
                y = -(a * 0 + c) / b
                if 0 <= y <= H:
                    points.append((0, int(y)))
            
            # Right edge (x=W)
            if abs(b) > 1e-6:
                y = -(a * W + c) / b
                if 0 <= y <= H:
                    points.append((W, int(y)))
            
            if len(points) >= 2:
                cv2.line(img, points[0], points[1], color, 2)

        draw_fitted_line(line_img, self.lineA, (0, 255, 255))  # green
        draw_fitted_line(line_img, self.lineB, (255, 255, 0))  # cyan

        print("Fitted lines drawn: green (boundary 1) and cyan (boundary 2)")

        self.proc_img = cv2_to_tk(line_img)
        self.proc_panel.config(image=self.proc_img)
        self.proc_panel.image = self.proc_img
        self.vars[6].set(1)

    def show_pose_extraction(self):
        if self.lineA is None or self.lineB is None:
            print("Run Segment Clustering first!")
            return

        # Use clustering visualization as background
        if not hasattr(self, 'clustering_img') or self.clustering_img is None:
            print("Run Segment Clustering first!")
            return
            
        # Copy clustering image to draw fitted lines on top
        line_img = self.clustering_img.copy()

        # Compute slot pose
        self.slot = lane_pair_to_slot_pose(self.lineA, self.lineB, line_img.shape, y_target_ratio=0.70)

        if self.slot is None:
            print("Could not compute slot pose from boundary lines")
            return

        corners = slot_pose_to_corners(self.slot)

        # Draw the slot box
        pts = np.array(corners, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(line_img, [pts], isClosed=True, color=(255, 0, 0), thickness=2)

        # Draw center point
        cx, cy = self.slot["center"]
        cv2.circle(line_img, (int(cx), int(cy)), 5, (0, 0, 255), -1)

        # Draw orientation arrow
        arrow_len = 30
        theta = self.slot["theta"]
        ex = int(cx + arrow_len * np.cos(theta))
        ey = int(cy + arrow_len * np.sin(theta))
        cv2.arrowedLine(line_img, (int(cx), int(cy)), (ex, ey), 
                       (0, 255, 255), 2, tipLength=0.3)

        print("=" * 60)
        print("PARKING SLOT DETECTED:")
        print(f"  Center: ({cx:.1f}, {cy:.1f}) pixels")
        print(f"  Orientation: {np.degrees(theta):.1f}°")
        print(f"  Width: {self.slot['width']:.1f} pixels")
        print(f"  Length: {self.slot['length']:.1f} pixels")
        print("=" * 60)

        # Add text overlay in right bottom corner with red color
        H, W = line_img.shape[:2]
        text_lines = [
            "PARKING SLOT DETECTED:",
            f"  Center: ({cx:.1f}, {cy:.1f}) pixels",
            f"  Orientation: {np.degrees(theta):.1f}°",
            f"  Width: {self.slot['width']:.1f} pixels",
            f"  Length: {self.slot['length']:.1f} pixels"
        ]
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        color = (0, 0, 255)  # Red in BGR
        line_height = 20
        
        # Calculate starting position (right bottom corner with padding)
        padding = 10
        y_start = H - padding - len(text_lines) * line_height
        
        for i, text in enumerate(text_lines):
            y_pos = y_start + i * line_height
            # Get text size to align from right
            (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
            x_pos = W - text_width - padding
            cv2.putText(line_img, text, (x_pos, y_pos), font, font_scale, color, thickness, cv2.LINE_AA)

        self.proc_img = cv2_to_tk(line_img)
        self.proc_panel.config(image=self.proc_img)
        self.proc_panel.image = self.proc_img
        self.vars[7].set(1)

    def process_all(self):
        if self.processor is None:
            print("Please load an image first!")
            return
        self.processor.process_all()
        self.show_edges()
        self.show_clustering()
        self.show_line_fitting()
        self.show_pose_extraction()
        for v in self.vars:
            v.set(1)

    # ---- parameter UI ----

    def add_param_controls(self):
        self.param_frame = Frame(self.root)
        self.param_frame.grid(row=2, column=0, columnspan=2, pady=10)

        self.param_vars = {}
        param_defs = [
            ('threshold',     10, 200),
            ('minLineLength', 10, 200),
            ('maxLineGap',    10, 100),
            ('angle_min',     15, 90),
            ('angle_max',     100, 180),
            ('min_length',    27, 200),
            ('angle_pair_th', 1, 30),
            ('min_pair_gap',  19, 300),
            ('max_pair_gap',  21, 400),
        ]

        for i, (name, minv, maxv) in enumerate(param_defs):
            # Always get defaults from HoughLaneDetector
            detector_defaults = HoughLaneDetector().params
            if self.processor is not None:
                default_val = self.processor.hough_params.get(name, detector_defaults.get(name, minv))
            else:
                default_val = detector_defaults.get(name, minv)
            
            var = tk.IntVar(value=int(default_val))
            self.param_vars[name] = var

            label = Label(self.param_frame, text=name)
            label.grid(row=0, column=i)
            slider = tk.Scale(self.param_frame, from_=minv, to=maxv,
                              orient=tk.HORIZONTAL, variable=var)
            slider.grid(row=1, column=i)

        self.update_btn = Button(self.param_frame, text="Update Parameters",
                                 command=self.update_params)
        self.update_btn.grid(row=2, column=0, columnspan=len(param_defs), pady=5)

    def update_params(self):
        for k, v in self.param_vars.items():
            val = v.get()
            # some params as float is okay, but int is fine too
            if k in ("angle_pair_th", "min_pair_gap", "max_pair_gap"):
                self.processor.hough_params[k] = float(val)
            else:
                self.processor.hough_params[k] = int(val)

        print("Updated parameters:", self.processor.hough_params)
        # Re-run LSD line detection with updated params (if applicable)
        # self.show_edges()

    def show_hough_lines(self):
        if self.processor is None:
            print("Please load an image first!")
            return
        edges = self.processor.step_edges(use_lsd=True)
        
        if self.use_auto_tune.get():
            # Auto-tuning mode: find best parameters
            print("\n=== AUTO-TUNING ===")
            
            # Suggest Hough threshold based on edge density
            auto_thr = auto_hough_threshold(edges)
            print(f"Suggested Hough threshold from edge density: {auto_thr}")
            
            # Run grid search to find best parameters
            base_params = self.processor.hough_params.copy()
            best_params, line_img, segments, score = auto_tune(
                HoughLaneDetector, edges, base_params, verbose=True
            )
            
            # Update processor params with best found
            self.processor.hough_params.update(best_params)
            
            # Update slider UI to reflect auto-tuned values
            for key in ['threshold', 'angle_pair_th']:
                if key in best_params and key in self.param_vars:
                    self.param_vars[key].set(int(best_params[key]))
            
            print(f"\n✓ Auto-tuned: {len(segments)} segments, score={score:.1f}")
            print(f"  → Sliders updated: threshold={best_params['threshold']}, angle_pair_th={best_params['angle_pair_th']:.1f}")
        else:
            # Manual mode: use current parameters
            params = self.processor.hough_params
            detector = HoughLaneDetector(params)
            line_img, segments = detector.detect(edges)
            print(f"\nDetected {len(segments)} lane segments (manual params)")

        # Store segments for next steps
        self.segments = segments

        self.proc_img = cv2_to_tk(line_img)
        self.proc_panel.config(image=self.proc_img)
        self.proc_panel.image = self.proc_img
        self.vars[5].set(1)
   
# ----------------------------------------------------------
# Main
# ----------------------------------------------------------

def main():
    root = Tk()
    app = App(root, img=None)
    root.mainloop()


if __name__ == "__main__":
    main()
