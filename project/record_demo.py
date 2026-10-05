"""Record one noisy, image-guided physical run without rerunning the whole suite."""
from pathlib import Path
import json
import imageio.v2 as imageio
from robot import Simulation
from vision import Calibration,process,dashboard
from validate import evaluate,placement

def record():
    out=Path(__file__).parent/'outputs'; sim=Simulation(); cal=Calibration(sim.model,sim.data)
    images,ds=process(sim.render(),'BLUE',cal,.08)
    assert len(ds)==2
    # Explicit recording-fixture instance choice: leftmost camera region.
    target=ds[0]; sim.plan(target.position,target.yaw)
    measurement=evaluate(sim,target,'blue'); bid=sim.model.body('blue').id; initial=sim.data.xpos[bid,2]; peak=[initial]
    with imageio.get_writer(out/'actual_demo.mp4',fps=20,codec='libx264',quality=7,macro_block_size=16) as writer:
        for _ in range(30): writer.append_data(dashboard(images,['BLUE | FOUND | Explicit camera instance 1 selected','Simulated noise: ON (8%) | Reachability: PASSED','Robot has not moved'],sim.render('scene')))
        def frame():
            peak[0]=max(peak[0],sim.data.xpos[bid,2])
            live,_=process(sim.render(),'BLUE',cal,.08)
            writer.append_data(dashboard(live,[f'Target color: BLUE | Locked image centre: {target.centre[0]:.1f}, {target.centre[1]:.1f}',f'World estimate: {target.position.round(3)} m | Yaw: {target.yaw:.1f} deg | Robot: {sim.state}','Finger contact physics; no box attachment or teleport'],sim.render('scene')))
        sim.callback=frame; sim.pick(target); sim.callback=None
        final,_=process(sim.render(),'BLUE',cal,.08)
        for _ in range(30): writer.append_data(dashboard(final,['DONE: selected blue box released inside tray','Robot returned to its observation pose','RGB, noisy, filtered, mask and overlay are final live images'],sim.render('scene')))
    measurement.update(placement(sim,'blue',initial,peak[0])); assert measurement['pick_place_success']
    (out/'recording_result.json').write_text(json.dumps(measurement,indent=2)); sim.close()
    print('Actual noisy-camera recording completed and physical placement verified.',flush=True)

if __name__=='__main__': record()
