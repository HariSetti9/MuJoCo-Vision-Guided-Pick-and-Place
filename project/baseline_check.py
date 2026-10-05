"""Run the unmodified upstream main with a bounded, offscreen viewer adapter."""
import sys, importlib.util, json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent.parent/'upstream/IK-With-MuJoCo-Solutions'
sys.path.insert(0,str(ROOT))
spec=importlib.util.spec_from_file_location('baseline',ROOT/'pick_n_place/dls_pick_n_place.py')
b=importlib.util.module_from_spec(spec); spec.loader.exec_module(b)
class Lock:
    def __enter__(self): pass
    def __exit__(self,*args): pass
class Viewer:
    def __init__(self,m,d): self.m=m; self.d=d; self.start=d.xpos[m.body('cube').id].copy(); self.peak=self.start[2]
    def lock(self): return Lock()
    def sync(self): self.peak=max(self.peak,self.d.xpos[self.m.body('cube').id,2])
    def is_running(self):
        if self.d.time<45: return True
        end=self.d.xpos[self.m.body('cube').id].copy(); tray=self.d.xpos[self.m.body('tray').id].copy()
        result=dict(start=self.start.tolist(),end=end.tolist(),tray=tray.tolist(),lift_m=float(self.peak-self.start[2]),in_tray=bool(np.linalg.norm(end[:2]-tray[:2])<.05))
        Path('project/outputs/baseline.json').write_text(json.dumps(result,indent=2)); print(result); return False
b.RNG_SEED=4
b.viewer.launch_passive=lambda m,d: Viewer(m,d)
b.main()
