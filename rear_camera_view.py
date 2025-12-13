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
from tkinter import Tk, Button, Label, Checkbutton, IntVar, Frame
from PIL import Image, ImageTk
import tkinter as tk

from hough_lane_detector import HoughLaneDetector  # <--- separate file

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

    def step_crop(self, manual=False):
        if manual:
            # Interactive ROI selection
            roi_box = cv2.selectROI("Select ROI", self.img, showCrosshair=True, fromCenter=False)
            cv2.destroyWindow("Select ROI")
            x, y, w, h = roi_box
            if w > 0 and h > 0:
                self.roi = self.img[y:y+h, x:x+w]
            else:
                # Fallback: bottom half
                self.roi = self.img[self.h//2:, :]
        else:
            # Crop bottom half by default
            y0 = self.h // 2
            self.roi = self.img[y0:, :]
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
    def __init__(self, root, img):
        self.root = root
        self.root.title("Rear Camera View Playground")
        self.processor = ImageProcessor(img)

        # Frames
        self.left_frame = Frame(self.root)
        self.left_frame.grid(row=0, column=0, padx=10, pady=10)
        self.right_frame = Frame(self.root)
        self.right_frame.grid(row=0, column=1, padx=10, pady=10)
        self.control_frame = Frame(self.root)
        self.control_frame.grid(row=1, column=0, columnspan=2, pady=10)

        # Original image
        self.orig_img = cv2_to_tk(img)
        self.orig_label = Label(self.left_frame, text="Original Image")
        self.orig_label.pack()
        self.orig_panel = Label(self.left_frame, image=self.orig_img)
        self.orig_panel.pack()

        # Processed image
        self.proc_label = Label(self.right_frame, text="Processed Image")
        self.proc_label.pack()
        self.proc_panel = Label(self.right_frame)
        self.proc_panel.pack()

        # Checklist buttons
        self.steps = [
            ("Manual Crop ROI", self.show_manual_crop),
            ("Convert to Grayscale", self.show_gray),
            ("Gaussian Blur", self.show_blur),
            ("Canny Edge Detection", self.show_edges),
            ("Hough Lane Detection", self.show_hough_lines),
        ]
        self.vars = [IntVar() for _ in self.steps]
        for i, (name, func) in enumerate(self.steps):
            btn = Button(self.control_frame, text=name, command=func)
            btn.grid(row=0, column=i, padx=5)
            chk = Checkbutton(self.control_frame, text="Done", variable=self.vars[i], state="disabled")
            chk.grid(row=1, column=i)

        # Button to process all
        self.all_btn = Button(self.control_frame, text="Process All Steps", command=self.process_all)
        self.all_btn.grid(row=2, column=0, columnspan=len(self.steps), pady=10)

        # Parameter sliders
        self.add_param_controls()

    # ---- step display methods ----

    def show_manual_crop(self):
        roi = self.processor.step_crop(manual=True)
        self.proc_img = cv2_to_tk(roi)
        self.proc_panel.config(image=self.proc_img)
        self.proc_panel.image = self.proc_img
        self.vars[0].set(1)

    def show_gray(self):
        gray = self.processor.step_gray()
        self.proc_img = cv2_to_tk(gray)
        self.proc_panel.config(image=self.proc_img)
        self.proc_panel.image = self.proc_img
        self.vars[1].set(1)

    def show_blur(self):
        blur = self.processor.step_blur()
        self.proc_img = cv2_to_tk(blur)
        self.proc_panel.config(image=self.proc_img)
        self.proc_panel.image = self.proc_img
        self.vars[2].set(1)

    def show_edges(self):
        edges = self.processor.step_edges()
        self.proc_img = cv2_to_tk(edges)
        self.proc_panel.config(image=self.proc_img)
        self.proc_panel.image = self.proc_img
        self.vars[3].set(1)

    def show_hough_lines(self):
        edges = self.processor.step_edges()
        params = self.processor.hough_params
        detector = HoughLaneDetector(params)
        line_img, segments = detector.detect(edges)

        self.proc_img = cv2_to_tk(line_img)
        self.proc_panel.config(image=self.proc_img)
        self.proc_panel.image = self.proc_img
        self.vars[4].set(1)

        print("Detected lane segments (x1, y1, x2, y2):")
        for seg in segments:
            print("  ", seg)

    def process_all(self):
        self.processor.process_all()
        self.show_hough_lines()
        for v in self.vars:
            v.set(1)

    # ---- parameter UI ----

    def add_param_controls(self):
        self.param_frame = Frame(self.root)
        self.param_frame.grid(row=2, column=0, columnspan=2, pady=10)

        self.param_vars = {}
        param_defs = [
            ('threshold',     1, 200),
            ('minLineLength', 1, 200),
            ('maxLineGap',    1, 100),
            ('angle_min',     0, 90),
            ('angle_max',     90, 180),
            ('min_length',    1, 200),
            ('angle_pair_th', 1, 30),
            ('min_pair_gap',  10, 300),
            ('max_pair_gap',  10, 400),
        ]

        for i, (name, minv, maxv) in enumerate(param_defs):
            default_val = self.processor.hough_params.get(name, minv)
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
    img = cv2.imread(TEST_IMAGE_PATH)
    if img is None:
        print(f"Error: Could not load image at {TEST_IMAGE_PATH}")
        return

    root = Tk()
    app = App(root, img)
    root.mainloop()


if __name__ == "__main__":
    main()
