"""
lane_prior.py

LSD-based line detection for rear camera parking assistance.
Uses Line Segment Detector for direct line extraction (no Hough needed).
Optimized for rear-view perspective cameras (not surround-view).
"""

import numpy as np
import cv2


def create_lane_prior(image, use_lsd=True):
    """
    Create lane prior using LSD (Line Segment Detector) for rear camera parking.
    LSD directly detects line segments - more accurate than Canny+Hough for parking lines.
    
    Args:
        image: BGR image (640x360)
        use_lsd: If True, use LSD; otherwise return empty mask
    
    Returns:
        lane_mask: Binary mask (uint8, 0 or 255) with detected line segments
    """
    H, W = image.shape[:2]
    
    if not use_lsd:
        return np.zeros((H, W), dtype=np.uint8)
    
    # Focus on bottom 60% where parking lines typically are
    roi_y_start = int(H * 0.4)
    roi = image[roi_y_start:, :]
    
    # Convert to grayscale
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # Create LSD detector (refine=0 for standard mode)
    lsd = cv2.createLineSegmentDetector(0)
    
    # Detect line segments directly
    lines = lsd.detect(gray)[0]
    
    # Create mask from detected line segments
    lane_mask_roi = np.zeros((roi.shape[0], roi.shape[1]), dtype=np.uint8)
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = map(int, line[0])
            cv2.line(lane_mask_roi, (x1, y1), (x2, y2), 255, 2)
        
        print(f"  LSD detected {len(lines)} line segments in bottom 60% ROI")
    else:
        print(f"  LSD: No line segments detected")
    
    # Restore to full image size
    lane_mask = np.zeros((H, W), dtype=np.uint8)
    lane_mask[roi_y_start:, :] = lane_mask_roi
    
    return lane_mask
