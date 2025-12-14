"""
rear_camera_view.py

Interactive playground for rear camera image processing:
- Manual ROI crop
- Grayscale
- Blur
- Canny edges
- Hough lane detection with parallel-pair filtering (via HoughLaneDetector)
"""

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

TEST_IMAGE_PATH = "test_parking_1.jpg"


def cv2_to_tk(img):
    """Convert OpenCV image (BGR or GRAY) to a Tkinter PhotoImage."""
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img)
    pil_img = pil_img.resize((400, 300))
    return ImageTk.PhotoImage(pil_img)


# ----------------------------------------------------------
# Image processor: crop → gray → blur → edges
# ----------------------------------------------------------

class ImageProcessor:
    def __init__(self, img):
        self.img = img
        self.h, self.w = img.shape[:2]
        self.roi = None
        self.gray = None
        self.blur = None
        self.edges = None

        # Initialize hough_params from detector defaults so they stay in sync
        self.hough_params = HoughLaneDetector().params.copy()

        # ROI config (tune these once, then keep fixed)
        self.roi_y0_ratio = 0.40  # start at 40% height -> bottom 60%
        self.roi_x0_ratio = 0.00  # full width; try 0.08 to cut left edge noise
        self.roi_x1_ratio = 1.00  # full width; try 0.92 to cut right edge noise

    def step_crop(self):
        """
        Fixed ROI crop to reduce perspective/crop sensitivity.
        """
        H, W = self.img.shape[:2]
        y0 = int(self.roi_y0_ratio * H)
        x0 = int(self.roi_x0_ratio * W)
        x1 = int(self.roi_x1_ratio * W)

        # clamp to valid range
        y0 = max(0, min(y0, H - 1))
        x0 = max(0, min(x0, W - 1))
        x1 = max(x0 + 1, min(x1, W))

        self.roi = self.img[y0:H, x0:x1]
        return self.roi

    def step_gray(self):
        if self.roi is None:
            self.step_crop()
        self.gray = cv2.cvtColor(self.roi, cv2.COLOR_BGR2GRAY)
        return self.gray

    def step_blur(self):
        if self.gray is None:
            self.step_gray()
        self.blur = cv2.GaussianBlur(self.gray, (5, 5), 0)
        return self.blur

    def step_edges(self):
        if self.blur is None:
            self.step_blur()
        self.edges = cv2.Canny(self.blur, 50, 150)
        return self.edges

    def process_all(self):
        self.step_crop()
        self.step_gray()
        self.step_blur()
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
        self.orig_label = Label(self.left_frame, text="Original Image")
        self.orig_label.pack()
        self.orig_panel = Label(self.left_frame)
        self.orig_panel.pack()
        
        # Load initial image if provided
        if img is not None:
            self.orig_img = cv2_to_tk(img)
            self.orig_panel.config(image=self.orig_img)
            self.orig_panel.image = self.orig_img

        # Processed image
        self.proc_label = Label(self.right_frame, text="Processed Image")
        self.proc_label.pack()
        self.proc_panel = Label(self.right_frame)
        self.proc_panel.pack()

        # State for slot detection steps
        self.segments = None
        self.lineA = None
        self.lineB = None
        self.slot = None

        # Checklist buttons
        self.steps_row1 = [
            ("Load Image", self.load_image),
            ("Convert to Grayscale", self.show_gray),
            ("Gaussian Blur", self.show_blur),
            ("Canny Edge Detection", self.show_edges),
            ("Hough Lane Detection", self.show_hough_lines),
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

        # Button to process all (far right of second row)
        self.all_btn = Button(self.control_frame, text="Process All Steps", command=self.process_all)
        self.all_btn.grid(row=2, column=len(self.steps_row1)-1, padx=5, sticky="e")

        # Parameter sliders
        self.add_param_controls()

    # ---- step display methods ----

    def load_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Parking Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp"),
                ("All files", "*.*")
            ],
            initialdir="."
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

    def show_edges(self):
        if self.processor is None:
            print("Please load an image first!")
            return
        edges = self.processor.step_edges()
        self.proc_img = cv2_to_tk(edges)
        self.proc_panel.config(image=self.proc_img)
        self.proc_panel.image = self.proc_img
        self.vars[3].set(1)

    def show_hough_lines(self):
        if self.processor is None:
            print("Please load an image first!")
            return
        edges = self.processor.step_edges()
        params = self.processor.hough_params
        detector = HoughLaneDetector(params)
        line_img, segments = detector.detect(edges)

        # Store segments for next steps
        self.segments = segments

        print(f"\nDetected {len(segments)} lane segments")

        self.proc_img = cv2_to_tk(line_img)
        self.proc_panel.config(image=self.proc_img)
        self.proc_panel.image = self.proc_img
        self.vars[4].set(1)

    def show_clustering(self):
        if self.segments is None or len(self.segments) < 2:
            print("Run Hough Lane Detection first!")
            return

        edges = self.processor.step_edges()
        line_img = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        # Cluster segments into two boundaries
        self.lineA, self.lineB = cluster_into_two_boundaries(self.segments)

        if self.lineA is None or self.lineB is None:
            print("Could not cluster segments into two boundary lines")
            return

        # Draw segments colored by cluster
        # Group 1: green, Group 2: cyan
        from slot_pose import seg_to_line_normal_form
        
        lines = []
        for s in self.segments:
            out = seg_to_line_normal_form(s)
            if out is None:
                continue
            a, b, c, ang = out
            if a < 0 or (abs(a) < 1e-6 and b < 0):
                a, b, c = -a, -b, -c
            lines.append((a, b, c, ang, s))

        if not lines:
            return

        lines.sort(key=lambda t: t[2])
        cs = np.array([t[2] for t in lines], dtype=np.float32)
        gaps = cs[1:] - cs[:-1]
        split_idx = int(np.argmax(gaps)) + 1

        group1 = lines[:split_idx]
        group2 = lines[split_idx:]

        for (_, _, _, _, s) in group1:
            x1, y1, x2, y2 = s
            cv2.line(line_img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # green

        for (_, _, _, _, s) in group2:
            x1, y1, x2, y2 = s
            cv2.line(line_img, (x1, y1), (x2, y2), (0, 255, 255), 2)  # cyan

        print(f"Clustered into 2 groups: {len(group1)} (green) and {len(group2)} (cyan) segments")

        self.proc_img = cv2_to_tk(line_img)
        self.proc_panel.config(image=self.proc_img)
        self.proc_panel.image = self.proc_img
        self.vars[5].set(1)

    def show_line_fitting(self):
        if self.lineA is None or self.lineB is None:
            print("Run Segment Clustering first!")
            return

        edges = self.processor.step_edges()
        line_img = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        # Draw the fitted infinite lines across the image
        H, W = edges.shape[:2]

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

        draw_fitted_line(line_img, self.lineA, (0, 255, 0))  # green
        draw_fitted_line(line_img, self.lineB, (0, 255, 255))  # cyan

        print("Fitted lines drawn: green (boundary 1) and cyan (boundary 2)")

        self.proc_img = cv2_to_tk(line_img)
        self.proc_panel.config(image=self.proc_img)
        self.proc_panel.image = self.proc_img
        self.vars[6].set(1)

    def show_pose_extraction(self):
        if self.lineA is None or self.lineB is None:
            print("Run Line Fitting first!")
            return

        edges = self.processor.step_edges()
        line_img = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        # Compute slot pose
        self.slot = lane_pair_to_slot_pose(self.lineA, self.lineB, edges.shape, y_target_ratio=0.70)

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

        self.proc_img = cv2_to_tk(line_img)
        self.proc_panel.config(image=self.proc_img)
        self.proc_panel.image = self.proc_img
        self.vars[7].set(1)

    def process_all(self):
        if self.processor is None:
            print("Please load an image first!")
            return
        self.processor.process_all()
        self.show_hough_lines()
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
        self.show_hough_lines()


# ----------------------------------------------------------
# Main
# ----------------------------------------------------------

def main():
    # Try to load default test image, but start without if not found
    img = cv2.imread(TEST_IMAGE_PATH)
    if img is None:
        print(f"Default test image not found at {TEST_IMAGE_PATH}")
        print("Please use 'Load Image' button to select an image")
        img = None

    root = Tk()
    app = App(root, img)
    root.mainloop()


if __name__ == "__main__":
    main()
