# utils/qp_velocity_control/qp_velocity_ctrl.py
# ------------------------------------------------------------------------------
# QPVelocityPlanner: QP-based IK planner for MuJoCo (OSQP)
#
# Requirements:
#   pip install osqp scipy
#
# Key features:
#  - Solves QP for joint increment Δq:
#       H = (2/dt^2) * (JᵀJ + λI)
#       c = -(1/dt)  * (Jᵀ xdot_d)
#    min 0.5 Δqᵀ H Δq + cᵀ Δq
#    s.t.  Δq_lb <= Δq <= Δq_ub
#          G Δq <= h   (optional)
#
#  - Joint limits are taken DIRECTLY from the MJCF joint range="min max"
#    (no reliance on model.jnt_limited); any "valid" range is treated as limited.
#
#  - In actuator_mode="position", integrates dq to q_ref and CLAMPS q_ref
#    to joint limits to avoid integrator windup.
#
#  - OSQP expects only upper-triangular Hessian (P). We use sparse.triu(P).
# ------------------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import mujoco

from utils.mj_velocity_control.mj_velocity_ctrl import JointVelocityController

import osqp
from scipy import sparse


@dataclass
class QPConstraints:
    """
    Optional linear inequality constraints:
        G * Δq <= h
    """
    G: Optional[np.ndarray] = None  # (m, n)
    h: Optional[np.ndarray] = None  # (m,)


class QPVelocityPlanner:
    """
    QP-based IK planner that solves for joint increments Δq using a quadratic program.

    Public API
    ----------
    reach_pose(target_pos, target_quat=None, ...)
        - target_quat is [x y z w] if provided
    track_twist(v_cart, w_cart=None, ...)
        - v_cart, w_cart are desired linear/angular velocities in base/world frame

    actuator_mode:
      - "torque"   : uses JointVelocityController to do vel-PD + gravity
      - "velocity" : writes desired joint velocities to data.ctrl
      - "position" : integrates dq into q_ref and writes q_ref to data.ctrl
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
        Quaternion logarithmic error (axis-angle 3-vector) between target q_t and current q_c.
        Convention: quaternions are [x y z w].
        """
        # q_err = q_t ⊗ q_c^{-1}
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
        model: mujoco.MjModel,
        data: mujoco.MjData,
        *,
        site_name: str = "right_center",
        damping: float = 1e-2,              # λ
        kd: float = 5.0,                    # only for torque mode
        gripper_cfg: list[dict] | None = None,
        actuator_mode: str = "torque",      # "torque" | "velocity" | "position"
        vel_limits: Optional[np.ndarray] = None,     # per arm joint (rad/s)
        qp_constraints: Optional[QPConstraints] = None,
        nullspace_weight: float = 0.0,
        q_null: Optional[np.ndarray] = None,
    ):
        self.model = model
        self.data = data
        self.site_name = site_name
        self.site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)

        self.lam = float(damping)

        self.actuator_mode = actuator_mode.lower().strip()
        if self.actuator_mode not in ("torque", "velocity", "position"):
            raise ValueError("actuator_mode must be one of: 'torque', 'velocity', 'position'")

        # Gripper actuator exclusion
        if gripper_cfg is not None:
            self.gripper_ids = {int(g["actuator_id"]) for g in gripper_cfg}
        else:
            self.gripper_ids = set()

        # Torque helper (gravity + velocity PD)
        self.ctrl = JointVelocityController(model, data, kd=kd, gripper_ids=self.gripper_ids)

        # ---- Build actuator->joint/dof/qpos mapping for arm actuators ----
        self.arm_act_ids: list[int] = []
        self.arm_joint_ids: list[int] = []
        self.arm_dof_ids: list[int] = []
        self.arm_qpos_ids: list[int] = []

        for act_id in range(model.nu):
            if act_id in self.gripper_ids:
                continue

            j_id = int(model.actuator_trnid[act_id, 0])
            if j_id < 0 or j_id >= model.njnt:
                continue

            dof_adr = int(model.jnt_dofadr[j_id])
            qpos_adr = int(model.jnt_qposadr[j_id])

            self.arm_act_ids.append(act_id)
            self.arm_joint_ids.append(j_id)
            self.arm_dof_ids.append(dof_adr)
            self.arm_qpos_ids.append(qpos_adr)

        self.n = len(self.arm_act_ids)
        if self.n == 0:
            raise RuntimeError("No arm actuators found (after excluding gripper).")

        # ---- Precompute per-arm joint ranges from MJCF joint range="min max" ----
        # Treat any meaningful finite range as a limit (do not rely on model.jnt_limited)
        self.arm_qmin = -np.inf * np.ones(self.n)
        self.arm_qmax = +np.inf * np.ones(self.n)

        for k, j_id in enumerate(self.arm_joint_ids):
            qmin, qmax = self.model.jnt_range[j_id]
            if np.isfinite(qmin) and np.isfinite(qmax) and (qmax - qmin) > 1e-9:
                self.arm_qmin[k] = float(qmin)
                self.arm_qmax[k] = float(qmax)

        # Optional velocity limits (rad/s) per arm joint
        self.vel_limits = None
        if vel_limits is not None:
            v = np.asarray(vel_limits, dtype=float).reshape(-1)
            if v.shape[0] != self.n:
                raise ValueError(f"vel_limits must have length {self.n}, got {v.shape[0]}")
            self.vel_limits = v

        # Optional GΔq<=h
        self.qp_constraints = qp_constraints if qp_constraints is not None else QPConstraints()

        # Optional nullspace bias
        self.nullspace_weight = float(nullspace_weight)
        self.q_null = None
        if q_null is not None:
            qn = np.asarray(q_null, dtype=float).reshape(-1)
            if qn.shape[0] != model.nq:
                raise ValueError(f"q_null must have length model.nq={model.nq}")
            self.q_null = qn.copy()

        # Jacobian buffers (full nv, we slice columns)
        self._jacp = np.zeros((3, model.nv))
        self._jacr = np.zeros((3, model.nv))

        # Position-mode reference
        self.q_ref = np.copy(self.data.qpos)

        # OSQP cache / warmstart
        self._osqp: Optional[osqp.OSQP] = None
        self._last_m: Optional[int] = None
        self._last_solution = np.zeros(self.n)

    # --------------------------- public methods --------------------------- #
    def set_nullspace_target(self, q_null: np.ndarray, weight: float = 0.0):
        qn = np.asarray(q_null, dtype=float).reshape(-1)
        if qn.shape[0] != self.model.nq:
            raise ValueError(f"q_null must have length model.nq={self.model.nq}")
        self.q_null = qn.copy()
        self.nullspace_weight = float(weight)

    def set_qp_inequality(self, G: Optional[np.ndarray], h: Optional[np.ndarray]):
        if G is None or h is None:
            self.qp_constraints = QPConstraints(None, None)
            return
        G = np.asarray(G, dtype=float)
        h = np.asarray(h, dtype=float).reshape(-1)
        if G.shape[0] != h.shape[0]:
            raise ValueError("G and h dimension mismatch")
        if G.shape[1] != self.n:
            raise ValueError(f"G must have {self.n} columns")
        self.qp_constraints = QPConstraints(G, h)

    def reach_pose(
        self,
        target_pos: np.ndarray,
        target_quat: Optional[np.ndarray] = None,  # [x y z w]
        pos_gain: float = 120.0,
        ori_gain: float = 12.0,
        dt: Optional[float] = None,
    ):
        if dt is None:
            dt = float(self.model.opt.timestep)

        ee_pos = self.data.site_xpos[self.site_id]
        ee_R = self.data.site_xmat[self.site_id].reshape(3, 3)
        pos_err = np.asarray(target_pos, dtype=float).reshape(3) - ee_pos

        if target_quat is None:
            cur_z = ee_R[:, 2]
            ori_err = np.cross(cur_z, np.array([0.0, 0.0, -1.0]))
        else:
            q_t = np.asarray(target_quat, dtype=float).reshape(4)
            q_t /= (np.linalg.norm(q_t) + 1e-12)

            q_wxyz = np.zeros(4)
            mujoco.mju_mat2Quat(q_wxyz, self.data.site_xmat[self.site_id])
            q_c = self._wxyz_to_xyzw(q_wxyz)

            if np.dot(q_t, q_c) < 0.0:
                q_t = -q_t

            ori_err = self._quat_log_error(q_t, q_c)

        xdot_d = np.concatenate([pos_gain * pos_err, ori_gain * ori_err])
        return self._solve_and_apply_qp(xdot_d, dt)

    def track_twist(
        self,
        v_cart: np.ndarray,
        w_cart: Optional[np.ndarray] = None,
        lin_gain: float = 1.0,
        ang_gain: float = 1.0,
        dt: Optional[float] = None,
    ):
        if dt is None:
            dt = float(self.model.opt.timestep)
        if w_cart is None:
            w_cart = np.zeros(3)

        v_cart = np.asarray(v_cart, dtype=float).reshape(3)
        w_cart = np.asarray(w_cart, dtype=float).reshape(3)

        xdot_d = np.concatenate([lin_gain * v_cart, ang_gain * w_cart])
        return self._solve_and_apply_qp(xdot_d, dt)

    # -------------------------- internal helpers -------------------------- #
    def _compute_reduced_jacobian(self) -> np.ndarray:
        mujoco.mj_jacSite(self.model, self.data, self._jacp, self._jacr, self.site_id)
        J_full = np.vstack([self._jacp, self._jacr])  # (6, nv)
        return J_full[:, self.arm_dof_ids]            # (6, n)

    def _box_bounds_deltaq(self, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Δq bounds from:
          - joint position limits using MJCF joint range="min max"
          - optional velocity limits (rad/s): |Δq| <= v_lim * dt
        """
        lb = -np.inf * np.ones(self.n)
        ub = +np.inf * np.ones(self.n)

        # Velocity limits
        if self.vel_limits is not None:
            dqmax = np.asarray(self.vel_limits, dtype=float)
            lb = np.maximum(lb, -dqmax * dt)
            ub = np.minimum(ub, +dqmax * dt)

        # Joint position limits -> qmin - q <= Δq <= qmax - q
        for k in range(self.n):
            qmin = self.arm_qmin[k]
            qmax = self.arm_qmax[k]
            if not (np.isfinite(qmin) and np.isfinite(qmax)):
                continue

            qcur = float(self.data.qpos[self.arm_qpos_ids[k]])
            lb = np.maximum(lb, qmin - qcur)
            ub = np.minimum(ub, qmax - qcur)

        bad = lb > ub
        if np.any(bad):
            lb[bad] = 0.0
            ub[bad] = 0.0

        return lb, ub

    def _build_qp(self, J: np.ndarray, xdot_d: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        H = (2/dt^2) * (JᵀJ + λI)
        c = -(1/dt)  * (Jᵀ xdot_d)
        """
        lamI = self.lam * np.eye(self.n)
        H = (2.0 / (dt * dt)) * (J.T @ J + lamI)
        c = -(1.0 / dt) * (J.T @ xdot_d)

        # Optional nullspace bias: 0.5*w*||Δq - (q_null - q)||^2
        if self.nullspace_weight > 0.0 and self.q_null is not None:
            w = self.nullspace_weight
            qcur = np.array([self.data.qpos[qid] for qid in self.arm_qpos_ids], dtype=float)
            qdes = np.array([self.q_null[qid] for qid in self.arm_qpos_ids], dtype=float)
            dq_null = (qdes - qcur)
            H = H + w * np.eye(self.n)
            c = c - w * dq_null

        H = 0.5 * (H + H.T)
        return H, c

    def _solve_qp_osqp(
        self,
        H: np.ndarray,
        c: np.ndarray,
        lb: np.ndarray,
        ub: np.ndarray,
        G: Optional[np.ndarray],
        h: Optional[np.ndarray],
    ) -> np.ndarray:
        """
        Solve:
          min 0.5 xᵀ H x + cᵀ x
          s.t. lb <= x <= ub
               G x <= h  (optional)

        OSQP expects P to contain only the upper triangular part.
        """
        n = self.n

        # OSQP form: min 0.5 xᵀ P x + qᵀ x  s.t. l <= A x <= u
        P = sparse.triu(sparse.csc_matrix(H), format="csc")  # IMPORTANT
        q = c.reshape(-1)

        I = sparse.eye(n, format="csc")
        A = I
        l = lb.copy()
        u = ub.copy()
        m = 0

        if G is not None and h is not None:
            G = np.asarray(G, dtype=float)
            h = np.asarray(h, dtype=float).reshape(-1)
            A = sparse.vstack([I, sparse.csc_matrix(G)], format="csc")
            l = np.concatenate([l, -np.inf * np.ones(G.shape[0])])
            u = np.concatenate([u, h])
            m = G.shape[0]

        # Setup/update OSQP
        if self._osqp is None or self._last_m != m:
            self._osqp = osqp.OSQP()
            self._osqp.setup(
                P=P, q=q, A=A, l=l, u=u,
                verbose=False,
                warm_start=True,
                polish=False,  # optional: avoid "Polishing not needed..." spam
            )
            self._last_m = m
        else:
            # Update numeric values (structure fixed if n and m fixed)
            self._osqp.update(Px=P.data, q=q, l=l, u=u)

        # Warm start
        self._osqp.warm_start(x=self._last_solution)

        res = self._osqp.solve()
        if res.info.status not in ("solved", "solved inaccurate"):
            return np.zeros(n)

        x = np.asarray(res.x, dtype=float).reshape(-1)
        self._last_solution = x.copy()
        return x

    def _solve_and_apply_qp(self, xdot_d: np.ndarray, dt: float):
        xdot_d = np.asarray(xdot_d, dtype=float).reshape(6)
        J = self._compute_reduced_jacobian()  # (6, n)

        H, c = self._build_qp(J, xdot_d, dt)
        lb, ub = self._box_bounds_deltaq(dt)

        G = self.qp_constraints.G
        h = self.qp_constraints.h

        delta_q = self._solve_qp_osqp(H, c, lb, ub, G, h)  # (n,)
        dq_arm = delta_q / dt

        return self._apply_actuator_mode(dq_arm)

    # --------------------- actuator command application -------------------- #
    def _apply_actuator_mode(self, dq_arm: np.ndarray):
        """
        dq_arm: (n,) joint velocities in arm actuator order.

        Returns:
          - torque mode   : (nu,) torque vector
          - velocity mode : (nu,) velocity ctrl
          - position mode : (nq,) q_ref
        """
        dq_arm = np.asarray(dq_arm, dtype=float).reshape(self.n)

        # Build per-actuator desired velocity target (nu,)
        dq_full = np.zeros(self.model.nu)
        for k, act_id in enumerate(self.arm_act_ids):
            dq_full[act_id] = dq_arm[k]

        # ---------------- torque mode ---------------- #
        if self.actuator_mode == "torque":
            self.ctrl.set_velocity_target(dq_full)
            tau = np.zeros(self.model.nu)

            for act_id in range(self.model.nu):
                if act_id in self.gripper_ids:
                    continue

                dof_i = self.ctrl.dof_indices[act_id]
                v_act = self.data.qvel[dof_i]
                v_tar = self.ctrl.v_targets[act_id]
                torque_d = -self.ctrl.kd[act_id] * (v_act - v_tar)

                tau[act_id] = self.data.qfrc_bias[dof_i] + torque_d
                self.data.ctrl[act_id] = tau[act_id]

            return tau

        # ---------------- velocity mode ---------------- #
        if self.actuator_mode == "velocity":
            vel_cmd = np.zeros(self.model.nu)
            for k, act_id in enumerate(self.arm_act_ids):
                vel_cmd[act_id] = dq_arm[k]

            for act_id in range(self.model.nu):
                if act_id in self.gripper_ids:
                    continue
                self.data.ctrl[act_id] = vel_cmd[act_id]

            return vel_cmd

        # ---------------- position mode ---------------- #
        if self.actuator_mode == "position":
            dt = float(self.model.opt.timestep)

            # integrate into q_ref (qpos space) and clamp to MJCF joint ranges
            for k, act_id in enumerate(self.arm_act_ids):
                qpos_i = self.arm_qpos_ids[k]
                self.q_ref[qpos_i] += dq_arm[k] * dt

                qmin = self.arm_qmin[k]
                qmax = self.arm_qmax[k]
                if np.isfinite(qmin) and np.isfinite(qmax):
                    self.q_ref[qpos_i] = float(np.clip(self.q_ref[qpos_i], qmin, qmax))

            # write desired positions to ctrl
            for k, act_id in enumerate(self.arm_act_ids):
                qpos_i = self.arm_qpos_ids[k]
                self.data.ctrl[act_id] = self.q_ref[qpos_i]

            return self.q_ref.copy()

        raise RuntimeError(f"Unsupported actuator_mode '{self.actuator_mode}'")
