# MuJoCo Vision-Guided Pick-and-Place

A classroom project where a simulated Panda robot finds a chosen-color box in its camera picture, estimates its position and angle, then picks it up and puts it in a tray.

The robot runs in MuJoCo on your computer. This project does not control a real robot.

## What is in this project?

- `project/` - the program, robot scene, setup files, measured test results, and example camera images.
- `presentation/` - the editable PowerPoint you supplied, the speaking script, and the completed demo video.
- `project/THIRD_PARTY.md` - credits and license details for reused robot files and workshop code.

## Install and run on Windows

You need a Windows desktop with working graphics, an internet connection for the first setup, and:

- 64-bit Python 3.12 from [python.org](https://www.python.org/downloads/). Keep **Tcl/Tk** selected in the installer so the dashboard window can open.
- Git for Windows from [git-scm.com](https://git-scm.com/download/win).

Open PowerShell and run these commands:

```powershell
git clone https://github.com/HariSetti9/MuJoCo-Vision-Guided-Pick-and-Place.git
cd MuJoCo-Vision-Guided-Pick-and-Place
Expand-Archive -Path .\MuJoCo_Vision_Guided_Pick_and_Place_Project.zip -DestinationPath .\project-files
cd .\project-files
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r project\requirements.lock.txt
.\.venv\Scripts\python.exe project\app.py
```

The last command opens the dashboard. After installation, you can start it by double-clicking `Run_Demo.cmd` from this folder. Keep the first PowerShell window open while using the dashboard so you can see any error messages.

## Dashboard guide

The large picture area has six panels:

- **RGB camera** - the actual color image rendered by the MuJoCo camera. When several boxes have the target color, click the box you want here.
- **Noisy image** - the same picture with optional simulated specks added.
- **Median-filtered image** - the image after reducing the simulated specks. It does not remove every kind of image noise.
- **Target mask** - white pixels show the selected color; black pixels show everything else. This panel is for viewing, not clicking to select a box.
- **Detection overlay** - shows the chosen box outline, centre, angle, and estimated position.
- **MuJoCo scene** - a view of the robot, table, boxes, and tray.

The status line above the pictures tells you whether the target was found, its estimated position and angle, and whether the robot is moving or has stopped. A message starting with **REJECTED** means the robot has not started moving because a check failed.

### What the top buttons do

- **Target color** - tells the camera which color to find: BLUE, RED, GREEN, or YELLOW. There are two blue boxes in the starting scene, so BLUE requires you to choose one of them in the RGB camera picture.
- **Simulated noise (8%)** - turns repeatable, computer-made image specks on or off. This noise is only for the image demonstration; it does not move or add objects to the scene.
- **Step-through** - pauses the explanation at each stage. Leave it checked for a classroom demo.
- **Record MP4** - records a new run to `project/outputs/live_demo.mp4`. It is off by default. The supplied completed recording is `presentation/Actual_MuJoCo_Demo.mp4`.
- **Detect / start stages** - captures a new camera picture and begins the detection steps.
- **Next stage** - reveals the next image-processing step, then checks whether the target can be reached.
- **Run pick-and-place** - starts the robot after a valid target is selected and the reachability check passes.
- **Reset** - returns the robot and all boxes to their starting positions. Use this before another example.
- **MuJoCo viewer** - opens a separate window showing the running robot. You can leave it closed and use the dashboard scene panel.
- **Pause / resume robot** - pauses or continues the robot during its movement.

### Place a box somewhere else

The second row of controls is for moving a box **before** asking the robot to pick it:

1. In **Place box**, choose which box to move: `blue`, `blue2`, `red`, or `green`.
2. Enter its centre position in **X (m)** and **Y (m)**, and its rotation in **Yaw (deg)**.
3. Click **Validate and place**. The app checks the position and robot reach before changing the scene. If it displays **REJECTED**, that box was not moved.

The tested area is X **0.38 to 0.67 m**, Y **-0.08 to 0.18 m**, and angle **-85 to +85 degrees**. Keep box centres at least **0.095 m** apart. These are tested limits for this scene, not the robot's entire possible work area. A box placed inside the tray is not a valid pickup target.

**Target color** and **Place box** do different things: target color chooses what the camera should find; Place box chooses which object to reposition.

## Step-by-step: pick one box

1. Click **Reset**. Check that the boxes are separated on the table, not collected in the tray.
2. Choose the **Target color**. For the two-blue-box example, choose **BLUE**.
3. Leave **Step-through** checked. Turn simulated noise on if you want to show the filter working. Click **Detect / start stages**.
4. Click **Next stage** to show camera capture, noise, filtering, and the color mask. Click it until the label says **Step 5 / 8: Box outline, centre and angle detected**.
5. If two boxes match, left-click directly on the box you want in the **RGB camera** panel. Do not click the mask or the detection overlay. The selected box is marked in the detection overlay. If only one box matches, the app selects it automatically.
6. Click **Next stage** once to show the estimated world position, then once more to check reachability. At **Step 7 / 8**, the status should say the reachability check passed.
7. Click **Run pick-and-place**. The robot approaches above the box, descends, closes its fingers, lifts the box, carries it to the tray, releases it, and returns to a safe pose.
8. Click **Reset** before trying another box.

The robot stays still until a box is detected, selected when needed, and accepted by the position and reach checks. If it is rejected, read the status message. For **Outside validated test workspace**, click **Reset** and choose a box that is on the table inside the marked work area. If the boxes remain piled by the tray after reset, close and restart the app, then check again before picking.

### Show a missing-target example

Choose **YELLOW** (there is no yellow box in the starting scene), then click **Detect / start stages** and continue to the detection step. The app should report **NOT FOUND** and the robot should remain still. Click **Reset** to restore the scene for the next example.

## Run the included tests

The included completed test run detected and placed **10 of 10** tested boxes. Average position difference was **0.200 cm** and average angle difference was **0.296 degrees**. The tests also checked that the program refuses missing or unsafe targets. These are measurements from this finite simulated test set, not a promise for every possible scene.

To run the tests again from PowerShell:

```powershell
.\.venv\Scripts\python.exe project\run_tests.py
```

The tests need the same working graphics support as the dashboard. They refresh files under `project/outputs/`; copy those files first if you want to keep the supplied results unchanged. The dashboard also has a **Run measured test mode** button, which runs the checks separately and shows a short result in the window.

The saved test table is `project/outputs/results.csv`. Other useful files in that folder include `summary.json`, `coordinate_log.json`, `workspace_checks.json`, and the actual camera, filtered-image, mask, and detection-overlay pictures.

## Presentation files

Open `presentation/MuJoCo_Vision_Guided_Pick_and_Place_Classroom.pptx` in PowerPoint. It is editable and has the demo video embedded on its video slide. `presentation/Actual_MuJoCo_Demo.mp4` is the same recording as a separate file. The recording shows a completed simulation; it is not a live camera feed.

`presentation/Classroom_Speaking_Script_Final.pdf` contains the speaking notes and a backup instruction for playing the recording if the live simulation runs poorly.

## Credits

This project reuses the Panda model and workshop code described in `project/THIRD_PARTY.md`. Keep the included credit and license files with the project. The camera dashboard, box-finding steps, controls, and classroom integration were made for this demonstration.


