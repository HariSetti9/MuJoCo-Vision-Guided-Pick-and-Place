import numpy as np


class quaternion_math:
    @staticmethod
    def rotmat_to_rpy(R: np.ndarray) -> np.ndarray:
        """
        Convert rotation matrix R (3x3) to roll-pitch-yaw (XYZ convention).
        Returns [roll, pitch, yaw] in radians.
        """
        R20 = np.clip(R[2, 0], -1.0, 1.0)
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = -np.arcsin(R20)
        yaw = np.arctan2(R[1, 0], R[0, 0])
        return np.array([roll, pitch, yaw])

    @staticmethod
    def wxyz_to_xyzw(q_wxyz: np.ndarray) -> np.ndarray:
        """Convert MuJoCo order [w x y z] → [x y z w]."""
        q = np.asarray(q_wxyz)
        return np.array([q[1], q[2], q[3], q[0]])

    @staticmethod
    def quat_log_error(q_t: np.ndarray, q_c: np.ndarray) -> np.ndarray:
        """
        Quaternion logarithmic error (axis-angle 3-vector) between target q_t
        and current q_c. Quaternions are [x y z w].
        """
        q_c_inv = np.array([-q_c[0], -q_c[1], -q_c[2], q_c[3]])
        x1, y1, z1, w1 = q_t
        x2, y2, z2, w2 = q_c_inv
        q_e = np.array([
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ])

        if q_e[3] < 0.0:
            q_e *= -1.0

        ang = 2.0 * np.arccos(np.clip(q_e[3], -1.0, 1.0))
        if ang < 1e-6:
            return np.zeros(3)
        axis = q_e[:3] / np.sin(ang / 2.0)
        return ang * axis