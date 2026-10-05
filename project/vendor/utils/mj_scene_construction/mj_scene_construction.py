import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict

import mujoco
import numpy as np
from mujoco.viewer import launch


@dataclass
class SceneInfo:
    cube_body: Optional[str]
    tray_body: Optional[str]
    table_body: Optional[str]
    cube_pos: Optional[np.ndarray]
    tray_pos: Optional[np.ndarray]
    table_top_z: float
    # --- NEW: extra cuboids for stacking examples ---
    blue_cuboid_body: Optional[str]
    green_cuboid_body: Optional[str]
    blue_cuboid_pos: Optional[np.ndarray]
    green_cuboid_pos: Optional[np.ndarray]


class SceneBuilder:
    """
    Minimal scene builder for Franka Panda scene.xml + simple environment:
      • optional table
      • tray with boundary walls
      • red cube (free joint)
      • blue cuboid (free joint)            <-- NEW
      • green cuboid (free joint)           <-- NEW
      • randomized tray + objects with min XY separation

    Builds an MjSpec from:
        <repo_root>/description/franka_emika_panda/scene.xml
    then programmatically adds table/tray/objects, compiles, and returns the MjModel.

    Notes:
      - Each object uses a FREE joint so you can move them at runtime if needed.
      - Randomization happens at build/compile time (unless you later edit qpos).
    """

    def __init__(
        self,
        include_table: bool = True,
        include_tray: bool = True,

        include_cube: bool = True,
        include_blue_cuboid: bool = True,     # <-- NEW
        include_green_cuboid: bool = True,    # <-- NEW

        randomize_tray: bool = True,
        randomize_cube: bool = True,
        randomize_blue_cuboid: bool = True,   # <-- NEW
        randomize_green_cuboid: bool = True,  # <-- NEW

        rng_seed: Optional[int] = None,
        robot_y: float = 0.0,

        min_tray_obj_dist: float = 0.10,      # tray <-> each object XY min distance
        min_obj_obj_dist: float = 0.08,       # object <-> object XY min distance (stacking prep)
    ):
        self.include_table = include_table
        self.include_tray = include_tray

        self.include_cube = include_cube
        self.include_blue_cuboid = include_blue_cuboid
        self.include_green_cuboid = include_green_cuboid

        self.randomize_tray = randomize_tray
        self.randomize_cube = randomize_cube
        self.randomize_blue_cuboid = randomize_blue_cuboid
        self.randomize_green_cuboid = randomize_green_cuboid

        self.robot_y = float(robot_y)

        self.min_tray_obj_dist = float(min_tray_obj_dist)
        self.min_obj_obj_dist = float(min_obj_obj_dist)

        self.rng = np.random.default_rng(rng_seed)

        # Resolve repo root and Panda scene path
        self.repo_root = self._find_repo_root()
        self.scene_xml_path = (
            Path(self.repo_root) / "description" / "franka_emika_panda" / "scene.xml"
        )

        # ----------------------------
        # Geometry (keep existing defaults)
        # ----------------------------
        self.table_center = np.array([0.55, self.robot_y, 0.20], dtype=float)
        self.table_extents = np.array([0.25, 0.35, 0.03], dtype=float)  # half-sizes
        self.table_top_z = float(self.table_center[2] + self.table_extents[2])

        # Red cube
        self.cube_half = 0.02
        self.cube_mass = 0.10

        # NEW: cuboids (dimensions chosen for stacking demos)
        # (These are "cuboids" not cubes: different x/y/z half-sizes.)
        self.blue_cuboid_half = np.array([0.03, 0.02, 0.02], dtype=float)   # 6x4x4 cm
        self.green_cuboid_half = np.array([0.025, 0.025, 0.015], dtype=float)  # 5x5x3 cm
        self.blue_cuboid_mass = 0.12
        self.green_cuboid_mass = 0.10

        # Tray base
        self.tray_extents = np.array([0.06, 0.06, 0.006], dtype=float)  # half-sizes
        self.tray_wall_thickness = 0.002
        self.tray_wall_height = 0.03

        # ----------------------------
        # Spawn bounds (world frame)
        # ----------------------------
        # Base XY region on table (shared among objects)
        self.object_xy_bounds = {
            "x": [self.table_center[0] - 0.18, self.table_center[0] + 0.18],
            "y": [self.robot_y - 0.22, self.robot_y + 0.22],
        }

        # Individual Z placement on top of table (each uses its own half-height)
        self.cube_spawn = {
            "x": self.object_xy_bounds["x"],
            "y": self.object_xy_bounds["y"],
            "z": [self.table_top_z + self.cube_half, self.table_top_z + self.cube_half],
        }

        self.blue_cuboid_spawn = {
            "x": self.object_xy_bounds["x"],
            "y": self.object_xy_bounds["y"],
            "z": [self.table_top_z + float(self.blue_cuboid_half[2]), self.table_top_z + float(self.blue_cuboid_half[2])],
        }

        self.green_cuboid_spawn = {
            "x": self.object_xy_bounds["x"],
            "y": self.object_xy_bounds["y"],
            "z": [self.table_top_z + float(self.green_cuboid_half[2]), self.table_top_z + float(self.green_cuboid_half[2])],
        }

        # Tray sits on top of table
        self.tray_spawn = {
            "x": [self.table_center[0] - 0.16, self.table_center[0] + 0.16],
            "y": [self.robot_y - 0.18, self.robot_y + 0.18],
            "z": [
                self.table_top_z + float(self.tray_extents[2] / 2),
                self.table_top_z + float(self.tray_extents[2] / 2),
            ],
        }

    # -------------------------
    # Public API
    # -------------------------
    def build(self) -> tuple[mujoco.MjModel, SceneInfo]:
        """
        Build and compile the Panda scene + environment.
        """
        if not self.scene_xml_path.exists():
            raise FileNotFoundError(f"Panda scene.xml not found at: {self.scene_xml_path}")

        spec = mujoco.MjSpec.from_file(str(self.scene_xml_path))
        spec.compiler.inertiafromgeom = True

        # Optional table
        table_name = None
        if self.include_table:
            table_name = "table"
            table = spec.worldbody.add_body(name=table_name, pos=self.table_center.tolist())
            table.add_geom(
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=self.table_extents.tolist(),
                rgba=[0.5, 0.4, 0.3, 1.0],
                contype=1,
                conaffinity=1,
            )

        # ----------------------------
        # Pick positions
        # ----------------------------
        tray_pos = None
        if self.include_tray:
            tray_pos = self._pick_position(self.tray_spawn, self.randomize_tray)

        # Objects requested
        obj_specs: Dict[str, dict] = {}

        if self.include_cube:
            obj_specs["cube"] = {
                "spawn": self.cube_spawn,
                "randomize": self.randomize_cube,
                "pos": None,
            }

        if self.include_blue_cuboid:
            obj_specs["blue_cuboid"] = {
                "spawn": self.blue_cuboid_spawn,
                "randomize": self.randomize_blue_cuboid,
                "pos": None,
            }

        if self.include_green_cuboid:
            obj_specs["green_cuboid"] = {
                "spawn": self.green_cuboid_spawn,
                "randomize": self.randomize_green_cuboid,
                "pos": None,
            }

        # Sample initial positions
        for name, s in obj_specs.items():
            s["pos"] = self._pick_position(s["spawn"], s["randomize"])

        # Enforce separation constraints
        tray_xy = None if tray_pos is None else np.array(tray_pos[:2], dtype=float)

        max_tries = 500
        for _ in range(max_tries):
            ok = True

            # 1) tray-object separation
            if tray_xy is not None:
                for s in obj_specs.values():
                    p_xy = np.array(s["pos"][:2], dtype=float)
                    if np.linalg.norm(p_xy - tray_xy) < self.min_tray_obj_dist:
                        ok = False
                        # resample the object if it is randomized, else resample tray if possible
                        if s["randomize"]:
                            s["pos"] = self._pick_position(s["spawn"], True)
                        elif self.randomize_tray:
                            tray_pos = self._pick_position(self.tray_spawn, True)
                            tray_xy = np.array(tray_pos[:2], dtype=float)
                        else:
                            # nothing can move; accept eventually
                            pass

            # 2) object-object separation
            names = list(obj_specs.keys())
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    pi = np.array(obj_specs[names[i]]["pos"][:2], dtype=float)
                    pj = np.array(obj_specs[names[j]]["pos"][:2], dtype=float)
                    if np.linalg.norm(pi - pj) < self.min_obj_obj_dist:
                        ok = False
                        # resample whichever is randomized; prefer resampling the latter
                        if obj_specs[names[j]]["randomize"]:
                            obj_specs[names[j]]["pos"] = self._pick_position(obj_specs[names[j]]["spawn"], True)
                        elif obj_specs[names[i]]["randomize"]:
                            obj_specs[names[i]]["pos"] = self._pick_position(obj_specs[names[i]]["spawn"], True)
                        elif self.randomize_tray and tray_pos is not None:
                            tray_pos = self._pick_position(self.tray_spawn, True)
                            tray_xy = np.array(tray_pos[:2], dtype=float)

            if ok:
                break

        # ----------------------------
        # Add tray + objects to spec
        # ----------------------------
        if self.include_tray and tray_pos is not None:
            self._add_tray(spec, tray_pos)

        if self.include_cube and obj_specs.get("cube", {}).get("pos") is not None:
            self._add_cube(spec, obj_specs["cube"]["pos"])

        if self.include_blue_cuboid and obj_specs.get("blue_cuboid", {}).get("pos") is not None:
            self._add_blue_cuboid(spec, obj_specs["blue_cuboid"]["pos"])

        if self.include_green_cuboid and obj_specs.get("green_cuboid", {}).get("pos") is not None:
            self._add_green_cuboid(spec, obj_specs["green_cuboid"]["pos"])

        # Compile
        model = spec.compile()

        # Contact tuning (optional but helps stability)
        model.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
        model.opt.impratio = 50
        model.opt.solver = mujoco.mjtSolver.mjSOL_PGS
        model.opt.iterations = 50
        model.opt.noslip_iterations = 100

        info = SceneInfo(
            cube_body="cube" if self.include_cube else None,
            tray_body="tray" if self.include_tray else None,
            table_body=table_name,
            cube_pos=None if not self.include_cube else np.asarray(obj_specs["cube"]["pos"], dtype=float),

            tray_pos=None if not self.include_tray else (None if tray_pos is None else np.asarray(tray_pos, dtype=float)),
            table_top_z=self.table_top_z,

            blue_cuboid_body="blue_cuboid" if self.include_blue_cuboid else None,
            green_cuboid_body="green_cuboid" if self.include_green_cuboid else None,
            blue_cuboid_pos=None if not self.include_blue_cuboid else np.asarray(obj_specs["blue_cuboid"]["pos"], dtype=float),
            green_cuboid_pos=None if not self.include_green_cuboid else np.asarray(obj_specs["green_cuboid"]["pos"], dtype=float),
        )
        return model, info

    # -------------------------
    # Scene elements
    # -------------------------
    def _add_tray(self, spec: mujoco.MjSpec, tray_pos: list[float]) -> None:
        tray = spec.worldbody.add_body(name="tray", pos=tray_pos)

        tray.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=self.tray_extents.tolist(),
            rgba=[0.7, 0.7, 0.7, 1.0],
            contype=1,
            conaffinity=1,
        )

        wall_x = [self.tray_wall_thickness, float(self.tray_extents[1]), self.tray_wall_height]
        wall_y = [float(self.tray_extents[0]), self.tray_wall_thickness, self.tray_wall_height]

        tray.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[float(self.tray_extents[0]), 0.0, self.tray_wall_height],
            size=wall_x,
            rgba=[0.3, 0.3, 0.3, 1.0],
            contype=1,
            conaffinity=1,
        )
        tray.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[-float(self.tray_extents[0]), 0.0, self.tray_wall_height],
            size=wall_x,
            rgba=[0.3, 0.3, 0.3, 1.0],
            contype=1,
            conaffinity=1,
        )

        tray.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[0.0, float(self.tray_extents[1]), self.tray_wall_height],
            size=wall_y,
            rgba=[0.3, 0.3, 0.3, 1.0],
            contype=1,
            conaffinity=1,
        )
        tray.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[0.0, -float(self.tray_extents[1]), self.tray_wall_height],
            size=wall_y,
            rgba=[0.3, 0.3, 0.3, 1.0],
            contype=1,
            conaffinity=1,
        )

    def _add_cube(self, spec: mujoco.MjSpec, cube_pos: list[float]) -> None:
        cube = spec.worldbody.add_body(name="cube", pos=cube_pos)
        cube.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[self.cube_half, self.cube_half, self.cube_half],
            rgba=[1.0, 0.0, 0.0, 1.0],
            mass=self.cube_mass,
            friction=[0.8, 0.01, 0.01],
            contype=1,
            conaffinity=1,
        )
        cube.add_joint(type=mujoco.mjtJoint.mjJNT_FREE, name="cube_free")

    # --- NEW: blue cuboid ---
    def _add_blue_cuboid(self, spec: mujoco.MjSpec, pos: list[float]) -> None:
        body = spec.worldbody.add_body(name="blue_cuboid", pos=pos)
        body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=self.blue_cuboid_half.tolist(),
            rgba=[0.1, 0.3, 1.0, 1.0],
            mass=self.blue_cuboid_mass,
            friction=[0.8, 0.01, 0.01],
            contype=1,
            conaffinity=1,
        )
        body.add_joint(type=mujoco.mjtJoint.mjJNT_FREE, name="blue_cuboid_free")

    # --- NEW: green cuboid ---
    def _add_green_cuboid(self, spec: mujoco.MjSpec, pos: list[float]) -> None:
        body = spec.worldbody.add_body(name="green_cuboid", pos=pos)
        body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=self.green_cuboid_half.tolist(),
            rgba=[0.1, 0.9, 0.2, 1.0],
            mass=self.green_cuboid_mass,
            friction=[0.8, 0.01, 0.01],
            contype=1,
            conaffinity=1,
        )
        body.add_joint(type=mujoco.mjtJoint.mjJNT_FREE, name="green_cuboid_free")

    # -------------------------
    # Sampling helpers
    # -------------------------
    def _pick_position(self, bounds: dict, randomize: bool) -> list[float]:
        if randomize:
            return [
                float(self.rng.uniform(*bounds["x"])),
                float(self.rng.uniform(*bounds["y"])),
                float(self.rng.uniform(*bounds["z"])),
            ]
        else:
            return [
                float(0.5 * (bounds["x"][0] + bounds["x"][1])),
                float(0.5 * (bounds["y"][0] + bounds["y"][1])),
                float(0.5 * (bounds["z"][0] + bounds["z"][1])),
            ]

    def _find_repo_root(self, target_dir: str = "description") -> str:
        current = Path(__file__).resolve().parent
        while True:
            if (current / target_dir).is_dir():
                return str(current)
            if current.parent == current:
                break
            current = current.parent
        raise FileNotFoundError(f"Could not find '{target_dir}' in any parent folder of {__file__}.")


# ------------------------------------------------------------------------------
# Demo usage (kept same; now shows extra cuboids too)
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    builder = SceneBuilder(
        include_table=True,
        include_tray=True,
        include_cube=True,
        include_blue_cuboid=True,
        include_green_cuboid=True,
        randomize_tray=True,
        randomize_cube=True,
        randomize_blue_cuboid=True,
        randomize_green_cuboid=True,
        rng_seed=0,
        robot_y=0.0,
        min_tray_obj_dist=0.12,
        min_obj_obj_dist=0.10,
    )

    model, info = builder.build()
    data = mujoco.MjData(model)

    key_name = "home"
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, key_name)

    if key_id != -1:
        mujoco.mj_resetDataKeyframe(model, data, key_id)
    else:
        if model.nkey > 0:
            print(f'[WARN] Keyframe "{key_name}" not found. Available keyframes:')
            for i in range(model.nkey):
                nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_KEY, i)
                print(f"  - {i}: {nm}")
        else:
            print("[WARN] No keyframes defined in this model.")
        mujoco.mj_resetData(model, data)

    mujoco.mj_forward(model, data)

    print("[INFO] SceneInfo:", info)

    viewer = launch(model, data)
    while viewer and viewer.is_running():
        viewer.render()
