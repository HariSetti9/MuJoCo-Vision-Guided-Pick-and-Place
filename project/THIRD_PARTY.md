# Attribution and licenses

This is a student demonstration assembled with coding assistance. Do not describe all code or robot assets as your own original work.

Both supplied repositories were inspected, with their README setup, XML model, DLS planner, joint controller, quaternion helpers and pick-and-place script:

- https://github.com/Debojit-D/IK-With-MuJoCo at `aba325dc4fbbeddb4671584d585c7719a4034b51`
- https://github.com/Debojit-D/IK-With-MuJoCo-Solutions at `25abca0164e8368c19b1705eaee401be85e83c2d`

The Solutions repository is the base. Its workshop Python utilities are copied unchanged to `vendor/utils` and carry Debojit Das's MIT license in `vendor/WORKSHOP_LICENSE.txt`. `robot.py` calls the upstream DLS planner's quaternion orientation-error helper, and adapts its damped least-squares Jacobian method into a bounded, joint-limited pose solver for preflight. The live movement uses the Panda XML position servos instead of the upstream velocity planner's continuously integrated references. This distinction must be explained accurately.

`vendor/panda` is copied from the Solutions repository. The Franka Emika Panda MuJoCo model comes from MuJoCo Menagerie / Google DeepMind and retains its Apache 2.0 license, README and change log. The robot meshes are reused, not student-created. Original source model: https://github.com/google-deepmind/mujoco_menagerie/tree/main/franka_emika_panda . See the model README for conversion provenance.

New integration: `scene.xml`, `vision.py`, `robot.py`, `validate.py`, `app.py`, `run_tests.py`, documentation and presentation content. These add calibration, image processing, color/instance selection, controls, bounded reachability, physical sequence, measured validation and classroom artifacts.

The supplied clones remain in the parent `upstream` folder. No tracked source edits were made there. `baseline_check.py` reads the upstream example and replaces only its viewer at runtime with a bounded adapter; it does not edit the example. Imported Python may create ordinary untracked bytecode caches.
