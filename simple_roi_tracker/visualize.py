import cv2
import numpy as np
from typing import Tuple, List
import time

from kalman_filter import KalmanFilter
from edge_aware_kalman_filter import EdgeAwareKalmanFilter

class InteractiveVisualizer:
    def __init__(self, img_size: Tuple[int, int] = (800, 1200), default_roi_size: Tuple[int, int] = (200, 300), margin_size: int = 150, margin_color: Tuple[int, int, int] = (55, 55, 55)):
        self.img_size = img_size
        self.default_roi_size: List[int, int] = list(default_roi_size)
        self.original_roi_size = default_roi_size
        self.margin_size = margin_size
        self.margin_color = margin_color
        self.current_roi = None
        self.rois_occlusion = True
        self.KF = KalmanFilter()
        cv2.namedWindow("Interactive Visualization")
        cv2.setMouseCallback("Interactive Visualization", self.mouse_event)

        self.current_time = time.time()
        self.last_time = self.current_time

    def mouse_event(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.current_roi = [x, y, self.default_roi_size[0], self.default_roi_size[1]]
            self.KF.init_state(self.current_roi)
            self.rois_occlusion = True
        elif event == cv2.EVENT_MOUSEMOVE and self.current_roi:
            self.current_roi[0], self.current_roi[1] = x, y
        elif event == cv2.EVENT_LBUTTONUP and self.current_roi:
            self.rois_occlusion = False

    def draw_roi(self, frame, roi, color=(0, 255, 0)):
        x, y, width, height = roi
        x1, y1 = x - width // 2, y - height // 2
        x2, y2 = x + width // 2, y + height // 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    def add_margin(self, frame):
        margin_color_bgr = self.margin_color[::-1]  # Convert to BGR if needed
        frame[:self.margin_size, :, :] = margin_color_bgr
        frame[-self.margin_size:, :, :] = margin_color_bgr
        frame[:, :self.margin_size, :] = margin_color_bgr
        frame[:, -self.margin_size:, :] = margin_color_bgr

    def get_measurement_roi(self):
        if not self.current_roi:
            return None

        x, y, width, height = self.current_roi
        x1 = max(self.margin_size, min(x - width // 2, self.img_size[1] - self.margin_size))
        y1 = max(self.margin_size, min(y - height // 2, self.img_size[0] - self.margin_size))
        x2 = min(self.img_size[1] - self.margin_size, max(x + width // 2, self.margin_size))
        y2 = min(self.img_size[0] - self.margin_size, max(y + height // 2, self.margin_size))
        # sanitize checker
        if x1 >= x2 or y1 >= y2:
            return None
        width_ = x2 - x1
        height_ = y2 - y1
        return [x1 + width_ // 2, y1 + height_ // 2, width_, height_]


    def key_control_roi_size(self, key):
        """control default roi size with key input
        w: increase roi width
        a: decrease roi width
        s: decrease roi height
        d: increase roi height
        r: reset roi size

        Args:
            key (_type_): _description_
        """
        if self.current_roi is None:
            return

        if key == ord('w') and self.current_roi:
            self.default_roi_size[1] += 10
        elif key == ord('s') and self.current_roi:
            self.default_roi_size[1] = max(10, self.default_roi_size[1] - 10)
        elif key == ord('a') and self.current_roi:
            self.default_roi_size[0] = max(10, self.default_roi_size[0] - 10)
        elif key == ord('d') and self.current_roi:
            self.default_roi_size[0] += 10
        elif key == ord('r') and self.current_roi:
            self.default_roi_size = list(self.original_roi_size.copy())

        self.current_roi[2] = self.default_roi_size[0] # height
        self.current_roi[3] = self.default_roi_size[1] # width

    def run(self):
        while True:
            self.current_time = time.time()
            dt = self.current_time - self.last_time
            # draw frame
            frame = np.zeros((self.img_size[0], self.img_size[1], 3), dtype=np.uint8)
            self.add_margin(frame)
            if self.current_roi:    
                self.draw_roi(frame, self.current_roi)

                # measurement and kalman filter estimation
                measurement_roi = self.get_measurement_roi() if not self.rois_occlusion else None
                if measurement_roi:
                    self.draw_roi(frame, measurement_roi, color=(0, 0, 255))
                kf_roi = self.KF.predict_and_update(dt, measurement_roi)
                self.draw_roi(frame, kf_roi, color=(255, 0, 0))

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            else:
                self.key_control_roi_size(key)

            cv2.imshow("Interactive Visualization", frame)
            self.last_time = self.current_time

        cv2.destroyAllWindows()

# 使用例
vis = InteractiveVisualizer()
vis.run()
