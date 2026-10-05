"""Ground truth is read only here, after image detection, for evaluation."""
import numpy as np
from vision import wrap_yaw

def evaluate(sim,detection,expected):
    bid=sim.model.body(expected).id
    truth=sim.data.xpos[bid].copy(); R=sim.data.xmat[bid].reshape(3,3)
    yaw=np.rad2deg(np.arctan2(R[1,0],R[0,0]))
    error=float(np.linalg.norm(detection.position-truth)*100)
    angle=abs(wrap_yaw(detection.yaw-yaw))
    return dict(detected_correctly=error<2,position_error_cm=error,angle_error_deg=angle,truth_xyz=truth.tolist(),truth_yaw=float(yaw),estimated_xyz=detection.position.tolist(),estimated_yaw=detection.yaw,image_centre=list(detection.centre),shape_validity=detection.validity)

def placement(sim,name,initial_z,peak):
    p=sim.data.xpos[sim.model.body(name).id]
    in_tray=abs(p[0]-.47)<.073 and abs(p[1]+.245)<.053 and .25<p[2]<.29
    open_fingers=bool(np.min(sim.data.qpos[7:9])>.03)
    return dict(pick_place_success=bool(in_tray and peak-initial_z>.08 and open_fingers),lift_m=float(peak-initial_z),final_xyz=p.tolist(),released=open_fingers)
