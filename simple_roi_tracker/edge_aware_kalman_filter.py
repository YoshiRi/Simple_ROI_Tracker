import numpy as np
from typing import Tuple, Optional

from kalman_filter import KalmanFilter

class EdgeAwareKalmanFilter(KalmanFilter):
    def __init__(self, dt:float = 0.1, process_noise_std: Tuple[float, float] = [0.5, 1.5], measurement_noise_std: float = 1.0,
                 edge_coordinates: Tuple[int, int, int, int] = [0, 0, 1200, 800]):
        """_summary_

        Args:
            dt (float, optional): _description_. Defaults to 0.1.
            process_noise_std (Tuple[float, float], optional): _description_. Defaults to [0.5, 1.5].
            measurement_noise_std (_type_, optional): _description_. Defaults to 1.0
            edge_coordinates:Tuple[int, int, int, int]=[0, 0, 1200, 800].
        """
        # initialize normal KalmanFilter
        super().__init__(dt, process_noise_std, measurement_noise_std)
        self.edge_coordinates = edge_coordinates
        self.process_noise_std = process_noise_std
        self.measurement_noise_std = measurement_noise_std
        self.define_measurement_matrix_for_corners()


    def calculate_roi_corners(self, measurement: Tuple[float, float, float, float]) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
        """calculate four roi corners from measurement

        Args:
            measurement (Tuple[float, float, float, float]): cx, cy, width, height
        """
        left_upper = (measurement[0] - measurement[2] // 2, measurement[1] - measurement[3] // 2)
        right_upper = (measurement[0] + measurement[2] // 2, measurement[1] - measurement[3] // 2)
        left_lower = (measurement[0] - measurement[2] // 2, measurement[1] + measurement[3] // 2)
        right_lower = (measurement[0] + measurement[2] // 2, measurement[1] + measurement[3] // 2)
        return [left_upper, right_upper, left_lower, right_lower]
    

    def check_valid_corners(self, corners: Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int], Tuple[int, int]]) -> Tuple[bool, bool, bool, bool]:
        """check if the corners are valid or not

        Args:
            corners (Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int], Tuple[int, int]]): _description_

        Returns:
            Tuple[bool, bool, bool, bool]: _description_
        """
        output = []
        for corner in corners:
            if not self.edge_coordinates[0] < corner[0] < self.edge_coordinates[2] or \
               not self.edge_coordinates[1] < corner[1] < self.edge_coordinates[3]:
                output.append(False)
            else:
                output.append(True)
        return tuple(output)
    
    def define_measurement_matrix_for_corners(self):
        """define measurement matrix for corners
            self.state : cx, cy, width, height, vx, vy, vwidth, vheight
            to left_upper, right_upper, left_lower, right_lower corners
        """
        left_uppers = np.array([[1., 0., -0.5, 0., 0., 0., 0., 0.],
                               [0., 1., 0., -0.5, 0., 0., 0., 0.]]).reshape(2, -1)
        right_uppers = np.array([[1., 0., 0.5, 0., 0., 0., 0., 0.],
                                [0., 1., 0., -0.5, 0., 0., 0., 0.]]).reshape(2, -1)
        left_lowers = np.array([[1., 0., -0.5, 0., 0., 0., 0., 0.],
                                 [0., 1., 0., 0.5, 0., 0., 0., 0.]]).reshape(2, -1)
        right_lowers = np.array([[1., 0., 0.5, 0., 0., 0., 0., 0.],
                                [0., 1., 0., 0.5, 0., 0., 0., 0.]]).reshape(2, -1)
        self.measurement_matrices_for_corners = [left_uppers, right_uppers, left_lowers, right_lowers]
    
    # override
    def update(self, measurement: Tuple[float, float, float, float]): 
        """_summary_

        Args:
            measurement ([type]): _description_
        """
        # measurement: (cx, cy, width, height)
        measurement_corners = self.calculate_roi_corners(measurement)
        valid_corners: Tuple[bool,bool,bool,bool] = self.check_valid_corners(measurement_corners)
        valid_corners_num = sum(valid_corners)

        if valid_corners_num == 0:
            # if all corners are invalid, do nothing
            return
        elif valid_corners_num == 1:
            # if there are only one valid corner, update only that corner
            measurement_matrix = self.measurement_matrices_for_corners[valid_corners.index(True)]
            measurement_cov = np.eye(2) * self.measurement_noise_std**2
            measurement_point = measurement_corners[valid_corners.index(True)]
        elif valid_corners_num == 2:
            # if there are only two valid corners, update only those corners
            measurement_matrix = np.vstack([self.measurement_matrices_for_corners[i] for i in range(4) if valid_corners[i]])
            measurement_cov = np.eye(4) * self.measurement_noise_std**2
            measurement_point = np.vstack([measurement_corners[i] for i in range(4) if valid_corners[i]])
        elif valid_corners_num == 4:
            measurement_matrix = self.measurement_matrix
            measurement_cov = self.measurement_noise_cov
            measurement_point = measurement
        else:
            # this should never happen
            raise ValueError("valid_corners_num == ", valid_corners_num, " is invalid.")

        # update 
        K = self.error_cov @ measurement_matrix.T @ np.linalg.inv(measurement_matrix @ self.error_cov @ measurement_matrix.T + measurement_cov)
        self.state = self.state + K @ (np.array(measurement_point).reshape(-1, 1) - measurement_matrix @ self.state)
        self.error_cov = self.error_cov - K @ measurement_matrix @ self.error_cov        