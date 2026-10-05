import numpy as np
import mujoco
from utils.mj_velocity_control.mj_velocity_ctrl import JointVelocityController


class DLSVelocityPlanner:
    """
    Damped Least-Squares (DLS) IK-based joint command planner.

    Supports three actuator modes:

    - actuator_mode="torque"   (default)
        Assumes torque-level actuators (e.g., <motor> or <general> behaving like motors).
        Internally uses JointVelocityController to do velocity-tracking PD + gravity
        compensation and writes resulting torques to data.ctrl.

    - actuator_mode="velocity"
        Assumes velocity servos (e.g., <velocity> shortcut or equivalent <general>).
        The solved joint velocities dq are written directly to data.ctrl (for the
        controlled actuators). No explicit gravity compensation here; the MuJoCo
        servo handles torque generation.

    - actuator_mode="position"
        Assumes position servos (e.g., Panda-style <general> where ctrl = q_des).
        The solved joint velocities dq are integrated to an internal q_ref using
        model.opt.timestep, and q_ref[dof_i] is written to data.ctrl[i].

    Public API
    ----------
    reach_pose(target_pos, target_quat=None)
        – Drive EE to `target_pos`; if `target_quat` is supplied (x y z w),
          also track the full orientation.  Otherwise only align local z
          with world −z.

    track_twist(v_cart, w_cart=None)
        – Map a desired 6-D twist to joint commands, then to actuator commands
          according to actuator_mode.

    All MuJoCo quaternions (stored as [w x y z]) are automatically converted
    to [x y z w] internally.
    """

    # --------------------------- static helpers --------------------------- #
    @staticmethod
    def _wxyz_to_xyzw(q_wxyz: np.ndarray) -> np.ndarray:
        """Convert MuJoCo order [w x y z] → [x y z w]."""
        q = np.asarray(q_wxyz)
        return np.array([q[1], q[2], q[3], q[0]])

    @staticmethod
    def _quat_log_error(q_t: np.ndarray, q_c: np.ndarray) -> np.ndarray:
        """
        Quaternion logarithmic error (axis-angle 3-vector) between target
        q_t and current q_c.  Convention: quaternions are [x y z w].
        """
        # q_err = q_t ⊗ q_c⁻¹
        q_c_inv = np.array([-q_c[0], -q_c[1], -q_c[2], q_c[3]])
        x1, y1, z1, w1 = q_t
        x2, y2, z2, w2 = q_c_inv
        q_e = np.array([
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ])

        # Hemisphere continuity
        if q_e[3] < 0.0:
            q_e *= -1.0

        ang = 2.0 * np.arccos(np.clip(q_e[3], -1.0, 1.0))
        if ang < 1e-6:
            return np.zeros(3)
        axis = q_e[:3] / np.sin(ang / 2.0)
        return ang * axis

    # ------------------------------ init ---------------------------------- #
    def __init__(
        self,
        model,
        data,
        kd: float = 5.0,
        site_name: str = "right_center",
        damping: float = 1e-2,
        gripper_cfg: list[dict] | None = None,
        for_multi: bool = False,
        actuator_mode: str = "torque",
    ):
        """
        Parameters
        ----------
        model, data : mujoco.MjModel, mujoco.MjData
        kd          : float
            Velocity error gain used by JointVelocityController (torque mode).
        site_name   : str
            End-effector site used for Jacobian and pose.
        damping     : float
            DLS damping lambda.
        gripper_cfg : list[dict] | None
            Each dict should contain "actuator_id" for gripper actuators
            to be excluded from arm control.
        for_multi   : bool
            If True, use the special slicing logic in _dls to map from
            nv → nu when multiple arms are present in one model.
        actuator_mode : {"torque", "velocity", "position"}
            - "torque"   : write torque commands (use with <motor>-like actuators)
            - "velocity" : write desired joint velocities to ctrl
            - "position" : integrate dq to q_ref and write desired joint positions
        """
        self.model = model
        self.data = data
        self.site_name = site_name
        self.damping = damping
        self.site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        self.for_multi = for_multi

        actuator_mode = actuator_mode.lower()
        if actuator_mode not in ("torque", "velocity", "position"):
            raise ValueError(
                f"Unknown actuator_mode='{actuator_mode}'. "
                "Use 'torque', 'velocity', or 'position'."
            )
        self.actuator_mode = actuator_mode

        # Extract actuator IDs from gripper_cfg
        if gripper_cfg is not None:
            gripper_ids = {g["actuator_id"] for g in gripper_cfg}
        else:
            gripper_ids = set()

        # Velocity-based torque controller (only used in torque mode)
        self.ctrl = JointVelocityController(model, data, kd=kd, gripper_ids=gripper_ids)

        # Re-usable buffers
        self._jacp = np.zeros((3, model.nv))
        self._jacr = np.zeros((3, model.nv))

        # Null-space control setup (disabled by default)
        self.use_nullspace = False
        self.q_null = np.zeros(model.nq)

        # For position-mode servos: internal reference joint configuration
        self.q_ref = np.copy(self.data.qpos)

    # --------------------------- public methods --------------------------- #
    def reach_pose(
        self,
        target_pos: np.ndarray,
        target_quat: np.ndarray | None = None,
        pos_gain: float = 120.0,
        ori_gain: float = 12.0,
    ):
        """
        Drive EE to target_pos; if target_quat (x y z w) is supplied,
        also track orientation.  Otherwise only align local z with world –z.

        Returns
        -------
        np.ndarray
            For actuator_mode="torque"   : applied torque vector (nu,)
            For actuator_mode="velocity" : applied velocity ctrl (nu,)
            For actuator_mode="position" : updated q_ref (nq,) after this step
        """
        # ── current EE pose ────────────────────────────────────────────────
        ee_pos = self.data.site_xpos[self.site_id]
        ee_R = self.data.site_xmat[self.site_id].reshape(3, 3)
        pos_err = target_pos - ee_pos

        # ── orientation error ──────────────────────────────────────────────
        if target_quat is None:
            # Align local Z to global –Z
            cur_z = ee_R[:, 2]
            ori_err = np.cross(cur_z, np.array([0.0, 0.0, -1.0]))
        else:
            # 1) normalise caller-supplied quaternion
            q_t = np.asarray(target_quat, float)
            q_t /= np.linalg.norm(q_t)

            # 2) current orientation (w x y z  →  x y z w)
            q_wxyz = np.zeros(4)
            mujoco.mju_mat2Quat(q_wxyz, self.data.site_xmat[self.site_id])
            q_c = self._wxyz_to_xyzw(q_wxyz)

            # 3) ensure both in same hemisphere
            if np.dot(q_t, q_c) < 0.0:
                q_t = -q_t

            # 4) logarithmic error
            ori_err = self._quat_log_error(q_t, q_c)

        # ── stack & solve ──────────────────────────────────────────────────
        task_err = np.concatenate([pos_gain * pos_err, ori_gain * ori_err])
        return self._error_to_actuator_cmd(task_err, pos_gain, ori_gain)

    def track_twist(
        self,
        v_cart: np.ndarray,
        w_cart: np.ndarray | None = None,
        lin_gain: float = 1.0,
        ang_gain: float = 1.0,
        damping: float | None = None,
    ):
        """
        Map desired twist to actuator commands (torque, velocity, or position
        depending on actuator_mode).

        Parameters
        ----------
        v_cart : (3,) ndarray
            Desired linear velocity in base frame.
        w_cart : (3,) ndarray or None
            Desired angular velocity; if None, zeros.
        lin_gain, ang_gain : float
            Optional scaling gains.
        damping : float or None
            Optional override of DLS lambda.

        Returns
        -------
        np.ndarray
            For actuator_mode="torque"   : applied torque vector (nu,)
            For actuator_mode="velocity" : applied velocity ctrl (nu,)
            For actuator_mode="position" : updated q_ref (nq,) after this step
        """
        if w_cart is None:
            w_cart = np.zeros(3)
        if damping is None:
            damping = self.damping

        twist = np.concatenate([lin_gain * v_cart, ang_gain * w_cart])
        return self._twist_to_actuator_cmd(twist, lin_gain, ang_gain, damping)

    # -------------------------- private helpers --------------------------- #
    def _compute_jac(self):
        mujoco.mj_jacSite(
            self.model, self.data, self._jacp, self._jacr, self.site_id
        )
        return self._jacp, self._jacr

    # -------- nullspace control -------- #
    def set_nullspace_target(self, q_null: np.ndarray, enable: bool = True):
        """
        Set the desired joint configuration for nullspace biasing.

        Parameters
        ----------
        q_null : (nq,) ndarray
            Desired joint configuration (full model qpos length).
        enable : bool
            If True, enables nullspace term. If False, only stores q_null but
            does not use it yet.
        """
        self.q_null = np.asarray(q_null, dtype=float).copy()
        self.use_nullspace = enable

    # -------- DLS solver -------- #
    def _dls(self, J, vec, lam):
        """
        Damped least-squares solution of J * dq ≈ vec.

        Returns dq projected to actuator space:
        - if for_multi is False: length nu (first nu components of dq_full)
        - if for_multi is True : uses active range heuristic to pack into nu
        """
        JT = J.T
        reg = lam * np.eye(J.shape[0])
        J_pinv = JT @ np.linalg.inv(J @ JT + reg)

        dq_task = J_pinv @ vec  # length nv

        if self.use_nullspace:
            I = np.eye(self.model.nv)
            dq_null = self.q_null[: self.model.nv] - self.data.qpos[: self.model.nv]
            dq_nullspace = (I - J_pinv @ J) @ dq_null
            dq_full = dq_task + dq_nullspace
        else:
            dq_full = dq_task

        # Map to actuator space (nu)
        if self.for_multi:
            # Find active indices with non-zero dq
            active_indices = np.nonzero(dq_full)[0]
            if len(active_indices) == 0:
                return np.zeros(self.model.nu)  # No active motion
            a = active_indices[0]
            b = active_indices[-1]
            active_dq = dq_full[a : b + 1]

            size = len(active_dq)
            dq_final = np.zeros(self.model.nu)

            if dq_full[0] == 0:
                dq_final[-size - 1 : -1] = active_dq
            else:
                dq_final[:size] = active_dq

            return dq_final
        else:
            # Assumes first nu dofs are actuated in joint order
            return dq_full[: self.model.nu]

    # -------- actuator command application -------- #
    def _apply_actuator_mode(self, dq: np.ndarray):
        """
        Core method that maps joint velocities dq (in actuator order) to
        data.ctrl depending on actuator_mode.

        Parameters
        ----------
        dq : (nu,) ndarray
            Joint velocities in actuator space.

        Returns
        -------
        np.ndarray
            - torque mode   : applied torques (nu,)
            - velocity mode : velocity control sent (nu,)
            - position mode : updated q_ref (nq,)
        """
        # ---------------- torque mode ---------------- #
        if self.actuator_mode == "torque":
            self.ctrl.set_velocity_target(dq)
            tau = np.zeros(self.model.nu)

            for i in range(self.ctrl.num_actuators):
                if i in self.ctrl.gripper_ids:
                    continue
                dof_i = self.ctrl.dof_indices[i]
                v_act = self.data.qvel[dof_i]
                v_tar = self.ctrl.v_targets[i]
                torque_d = -self.ctrl.kd[i] * (v_act - v_tar)
                tau[i] = self.data.qfrc_bias[dof_i] + torque_d
                self.data.ctrl[i] = tau[i]

            return tau

        # ---------------- velocity mode ---------------- #
        elif self.actuator_mode == "velocity":
            # Direct mapping: dq is desired joint velocity per actuator
            vel_cmd = np.zeros(self.model.nu)
            vel_cmd[: len(dq)] = dq

            for i in range(self.ctrl.num_actuators):
                if i in self.ctrl.gripper_ids:
                    continue
                # We assume "velocity servo": ctrl[i] = desired joint velocity
                self.data.ctrl[i] = vel_cmd[i]

            return vel_cmd

        # ---------------- position mode ---------------- #
        elif self.actuator_mode == "position":
            # Integrate dq into q_ref at the DOF level using model timestep
            dt = float(self.model.opt.timestep)

            # Update only actuated DOFs according to dq[actuator_index]
            for i in range(self.ctrl.num_actuators):
                if i in self.ctrl.gripper_ids:
                    continue
                dof_i = self.ctrl.dof_indices[i]
                self.q_ref[dof_i] += dq[i] * dt

            # Write desired positions into ctrl (position servo behaviour)
            for i in range(self.ctrl.num_actuators):
                if i in self.ctrl.gripper_ids:
                    continue
                dof_i = self.ctrl.dof_indices[i]
                self.data.ctrl[i] = self.q_ref[dof_i]

            return self.q_ref.copy()

        else:
            # Should never reach here due to validation in __init__
            raise RuntimeError(f"Unsupported actuator_mode '{self.actuator_mode}'")

    # -------- mapping from task error / twist -------- #
    def _error_to_actuator_cmd(self, err, lin_gain, ang_gain, lam=None):
        if lam is None:
            lam = self.damping
        jacp, jacr = self._compute_jac()
        J = np.vstack([lin_gain * jacp, ang_gain * jacr])
        dq = self._dls(J, err, lam)
        return self._apply_actuator_mode(dq)

    def _twist_to_actuator_cmd(self, twist, lin_gain, ang_gain, lam):
        jacp, jacr = self._compute_jac()
        J = np.vstack([lin_gain * jacp, ang_gain * jacr])
        dq = self._dls(J, twist, lam)
        return self._apply_actuator_mode(dq)

    # ---------------- convenience wrappers ---------------- #
    def get_torque_command(self, target_pos, target_quat=None):
        """
        Public method kept for backward compatibility.

        For actuator_mode="torque"   : returns torque vector (nu,)
        For actuator_mode="velocity" : returns velocity ctrl (nu,)
        For actuator_mode="position" : returns q_ref (nq,)
        """
        return self.reach_pose(target_pos, target_quat=target_quat)

    def get_torque_for_cartesian_velocity(
        self, v_cart, w_cart=None, damping=None, ori_gain=1.0
    ):
        """
        Public method kept for backward compatibility.

        For actuator_mode="torque"   : returns torque vector (nu,)
        For actuator_mode="velocity" : returns velocity ctrl (nu,)
        For actuator_mode="position" : returns q_ref (nq,)
        """
        return self.track_twist(
            v_cart, w_cart=w_cart, ang_gain=ori_gain, damping=damping
        )
