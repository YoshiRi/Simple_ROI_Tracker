import numpy as np
from typing import Tuple

class KalmanFilter:
    def __init__(self, dt:float = 0.1, process_noise_std: float = 0.5, measurement_noise_std: float = 1.0):
        # 状態量 (x, y, width, height)
        self.state = np.zeros(4)  

        # 制御行列 (ここでは制御入力はないと仮定)
        self.control_matrix = None  

        # 移動行列
        self.transition_matrix = np.array([[1, 0, dt, 0],
                                           [0, 1, 0, dt],
                                           [0, 0, 1, 0],
                                           [0, 0, 0, 1]])

        # 観測行列
        self.measurement_matrix = np.eye(4)

        # プロセスノイズの共分散
        self.process_noise_cov = np.eye(4) * process_noise_std**2

        # 観測ノイズの共分散
        self.measurement_noise_cov = np.eye(4) * measurement_noise_std**2

        # 誤差共分散
        self.error_cov = 100.0 * np.eye(4)

    def init_state(self, measurement: Tuple[float, float, float, float]):
        self.state = np.array(measurement)

    def predict(self):
        # 状態予測
        self.state = np.dot(self.transition_matrix, self.state)
        # 誤差共分散の更新
        self.error_cov = np.dot(np.dot(self.transition_matrix, self.error_cov), self.transition_matrix.T) + self.process_noise_cov

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
        K = np.dot(np.dot(self.error_cov, self.measurement_matrix.T), np.linalg.inv(np.dot(np.dot(self.measurement_matrix, self.error_cov), self.measurement_matrix.T) + self.measurement_noise_cov))
        # 状態の更新
        self.state = self.state + np.dot(K, (measurement - np.dot(self.measurement_matrix, self.state)))
        # 誤差共分散の更新
        self.error_cov = self.error_cov - np.dot(np.dot(K, self.measurement_matrix), self.error_cov)

    def predict_and_update(self, dt: float, measurement: Tuple[float, float, float, float]):
        self.transition_matrix = np.array([[1, 0, dt, 0],
                                           [0, 1, 0, dt],
                                           [0, 0, 1, 0],
                                           [0, 0, 0, 1]])
        self.predict()
        self.update(measurement)

        output_state: Tuple[int, int, int, int] = tuple(self.state.astype(int))
        return output_state