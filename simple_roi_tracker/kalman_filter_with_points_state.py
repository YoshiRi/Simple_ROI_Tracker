"""Kalman filter with points state

Objective:
- Normal kalman filter with upperleft and lowerright coordinates as its state
- Assume constant velocity model

Current status:
- worked but not different from normal kalman filter
"""

import numpy as np
from typing import Tuple, Optional

from kalman_filter import KalmanFilter

class KalmanFilterWithPointsState(KalmanFilter):
    def __init__(self, dt:float = 0.1, process_noise_std: Tuple[float, float] = [0.5, 1.5], measurement_noise_std: float = 1.0):
        # initialize normal KalmanFilter
        super().__init__(dt, process_noise_std, measurement_noise_std)

    def state_conversion(self, input_state: Tuple[float, float, float, float], convert:str = "forward") -> Tuple[float, float, float, float]:
        """convert state from upperleft and lowerright coordinates to center and shape

        Args:
            input_state (Tuple[float, float, float, float]): upperleft and lowerright coordinates

        Returns:
            Tuple[float, float, float, float]: center and shape
        """
        # upperleft and lowerright coordinates to center and shape
        if convert == "forward":
            x1, y1, x2, y2 = input_state
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            width = x2 - x1
            height = y2 - y1
            return cx, cy, width, height
        elif convert == "backward":
            cx, cy, width, height = input_state
            x1 = cx - width // 2
            y1 = cy - height // 2
            x2 = cx + width // 2
            y2 = cy + height // 2
            return x1, y1, x2, y2
        
    def init_state(self, measurement: Tuple[float, float, float, float]):
        measurement_in_pts = self.state_conversion(measurement, "backward")
        self.state[0:4, 0] = np.array(measurement_in_pts)[0:4]

    def update(self, measurement): # override
        # measurement: (x, y, width, height)
        measurement_in_pts = self.state_conversion(measurement, "backward") # x1, y1, x2, y2

        # カルマンゲインの計算
        K = self.error_cov @ self.measurement_matrix.T @ np.linalg.inv(self.measurement_matrix @ self.error_cov @ self.measurement_matrix.T + self.measurement_noise_cov)
        # 状態の更新
        self.state = self.state + K @ (np.array(measurement_in_pts).reshape(-1, 1) - self.measurement_matrix @ self.state)
        # 誤差共分散の更新
        self.error_cov = self.error_cov - K @ self.measurement_matrix @ self.error_cov

    def to_output_state(self)->Tuple[int, int, int, int]:
        tmp = tuple(self.state[0:4, 0])
        output_float = self.state_conversion(tmp, "forward") # to center and shape
        return tuple(map(int, output_float)) # to int