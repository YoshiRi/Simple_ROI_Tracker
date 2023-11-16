import numpy as np
from typing import Tuple, Optional

class KalmanFilter:
    def __init__(self, dt:float = 0.1, process_noise_std: Tuple[float, float] = [0.5, 1.5], measurement_noise_std: float = 1.0):
        # 状態量 (x, y, width, height, dx, dy, dwidth, dheight)
        self.state = np.zeros(8).reshape(-1, 1)

        # 制御行列 (ここでは制御入力はないと仮定)
        self.control_matrix = None  

        # 移動行列
        self.transition_matrix = np.eye(8)
        self.transition_matrix[0, 4] = dt
        self.transition_matrix[1, 5] = dt
        self.transition_matrix[2, 6] = dt
        self.transition_matrix[3, 7] = dt

        # 観測行列
        self.measurement_matrix = np.zeros((4, 8))
        self.measurement_matrix[0:4, 0:4] = np.eye(4)

        # プロセスノイズの共分散
        self.process_noise_cov = np.eye(8) 
        self.process_noise_cov[0:4, 0:4] *= process_noise_std[0]**2
        self.process_noise_cov[4:8, 4:8] *= process_noise_std[1]**2

        # 観測ノイズの共分散
        self.measurement_noise_cov = np.eye(4) * measurement_noise_std**2

        # 誤差共分散
        self.error_cov = 100.0 * np.eye(8)

    def init_state(self, measurement: Tuple[float, float, float, float]):
        self.state[0:4, 0] = np.array(measurement)[0:4]

    def predict(self):
        # 状態予測
        self.state = self.transition_matrix @ self.state
        # 誤差共分散の更新
        self.error_cov = self.transition_matrix @ self.error_cov @ self.transition_matrix.T + self.process_noise_cov

    def project(self):
        x, y, width, height = self.state
        top_left = (x - width // 2, y - height // 2)
        top_right = (x + width // 2, y - height // 2)
        bottom_left = (x - width // 2, y + height // 2)
        bottom_right = (x + width // 2, y + height // 2)
        return [top_left, top_right, bottom_left, bottom_right]

    def update(self, measurement):
        # measurement: (x, y, width, height)
        # カルマンゲインの計算
        K = self.error_cov @ self.measurement_matrix.T @ np.linalg.inv(self.measurement_matrix @ self.error_cov @ self.measurement_matrix.T + self.measurement_noise_cov)
        # 状態の更新
        self.state = self.state + K @ (np.array(measurement).reshape(-1, 1) - self.measurement_matrix @ self.state)
        # 誤差共分散の更新
        self.error_cov = self.error_cov - K @ self.measurement_matrix @ self.error_cov

    def predict_and_update(self, dt: float, measurement: Optional[Tuple[float, float, float, float]]):
        self.transition_matrix = np.eye(8)
        self.transition_matrix[0, 4] = dt
        self.transition_matrix[1, 5] = dt
        self.transition_matrix[2, 6] = dt
        self.transition_matrix[3, 7] = dt

        self.predict()
        if measurement:
            self.update(measurement)

        output_state: Tuple[int, int, int, int] = tuple(self.state.astype(int)[0:4, 0])
        return output_state