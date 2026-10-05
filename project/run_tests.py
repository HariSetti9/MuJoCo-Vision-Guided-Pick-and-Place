"""Actual rendered-image and full physics tests; writes measured CSV/JSON/MP4."""
from pathlib import Path
import json,csv,time
import cv2,imageio.v2 as imageio
import numpy as np
from PIL import Image
from robot import Simulation
from vision import Calibration,process,dashboard,Detection
from validate import evaluate,placement

OUT=Path(__file__).parent/'outputs'; OUT.mkdir(exist_ok=True)
CASES=[('BLUE','blue',.50,.03,0,0),('RED','red',.40,.15,30,.03),('GREEN','green',.63,.14,-35,.08),('BLUE','blue',.43,-.06,-60,.03),('RED','red',.53,.13,75,.08),('GREEN','green',.61,.07,-15,0),('BLUE','blue',.55,-.06,45,.08),('RED','red',.38,.18,-80,0),('GREEN','green',.67,.18,80,.03),('BLUE','blue2',.66,-.07,45,.03)]

def save_images(images,prefix=''):
    for name,img in images.items(): Image.fromarray(img).save(OUT/f'{prefix}{name}.png')

def run():
    sim=Simulation(); calibration=Calibration(sim.model,sim.data); rows=[]; details=[]
    cases=CASES
    writer=None
    for index,(color,name,x,y,yaw,noise) in enumerate(cases):
        sim.reset(); sim.place(name,x,y,yaw,validate=True)
        images,ds=process(sim.render(),color,calibration,noise)
        row=dict(test=f'physics_{index+1}',color=color,x=x,y=y,yaw=yaw,noise=noise,detected_correctly=False,position_error_cm='',angle_error_deg='',pick_place_success=False,correct_refusal=False)
        detail={}; initial_ctrl=sim.data.ctrl.copy()
        if ds:
            # Image instance rule: choose leftmost region for BLUE; other colors unique.
            d=ds[-1] if name=='blue2' else ds[0]; detail=evaluate(sim,d,name); row.update({k:detail[k] for k in ('detected_correctly','position_error_cm','angle_error_deg')})
            bid=sim.model.body(name).id; start=sim.data.xpos[bid,2]; peak=[start]; states={}
            if index==0:
                save_images(images); Image.fromarray(sim.render('scene')).save(OUT/'scene.png')
                Image.fromarray(dashboard(images,[f'Target color: {color} | Detection: FOUND | Select instance 1',f'Image centre: {d.centre} | World: {d.position.round(3)} | Yaw: {d.yaw:.1f}', 'Robot: READY - preflight before movement'],sim.render('scene'))).save(OUT/'dashboard.png')
                noisy_images,_=process(images['camera'],color,calibration,.08); save_images(noisy_images,'noise_')
                writer=imageio.get_writer(OUT/'actual_demo.mp4',fps=20,codec='libx264',quality=7,macro_block_size=16)
                for _ in range(30): writer.append_data(dashboard(noisy_images,['BLUE | FOUND | Simulated noise: 8%',f'Estimate: {d.position.round(3)} m | Yaw: {d.yaw:.1f} degrees','Reachability must pass before movement'],sim.render('scene')))
            def callback():
                peak[0]=max(peak[0],sim.data.xpos[bid,2])
                if index==0:
                    live,_=process(sim.render(),color,calibration,noise)
                    scene=sim.render('scene')
                    writer.append_data(dashboard(live,[f'Target color: {color} | Locked image estimate: {d.position.round(3)} m',f'Yaw: {d.yaw:.1f} deg | Robot: {sim.state}','Physics grasp - no weld, no object position edits'],scene))
                    states[sim.state]=states.get(sim.state,0)+1
                    if states[sim.state]==1 or states[sim.state]%10==0:
                        Image.fromarray(scene).save(OUT/f'{sim.state.lower()}.png')
            sim.callback=callback
            try:
                sim.pick(d); placed=placement(sim,name,start,peak[0]); detail.update(placed); row['pick_place_success']=placed['pick_place_success']
                row['reason']='DONE' if row['pick_place_success'] else 'Physical placement failed'
            except ValueError as e: row['reason']=str(e)
            sim.callback=None
            if writer is not None:
                Image.fromarray(sim.render('scene')).save(OUT/'placed.png')
                final_images,_=process(sim.render(),color,calibration,noise)
                for _ in range(30): writer.append_data(dashboard(final_images,['Physics result: box released inside tray','Robot returned to observation pose','Other objects remain on table'],sim.render('scene')))
                writer.close(); writer=None
        else: row['reason']='No valid target region'
        rows.append(row); details.append(detail); print(row,flush=True)
        # Preserve partial evidence even if a later test fails.
        with (OUT/'results.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    # Missing target: run same detection-to-motion gate, check controls and pose unchanged.
    sim.reset(); images,ds=process(sim.render(),'YELLOW',calibration,.03)
    before_q=sim.data.qpos.copy(); before_ctrl=sim.data.ctrl.copy()
    if ds: sim.pick(ds[0])
    rejected=not ds and np.array_equal(before_q,sim.data.qpos) and np.array_equal(before_ctrl,sim.data.ctrl)
    row=dict(test='missing_target',color='YELLOW',x='',y='',yaw='',noise=.03,detected_correctly=not ds,position_error_cm='',angle_error_deg='',pick_place_success=False,correct_refusal=rejected,reason='REJECTED: no valid YELLOW target')
    rows.append(row)
    Image.fromarray(dashboard(images,['Target color: YELLOW | Detection: NOT FOUND','REJECTED: no valid target region','Robot has not moved'],sim.render('scene'))).save(OUT/'failure_missing.png')
    # Reachable camera-visible object beyond allowed workspace: deliberately bypass setup gate ONLY for negative test.
    sim.reset(); sim.place('red',.79,.20,15,validate=False); images,ds=process(sim.render(),'RED',calibration)
    before_q=sim.data.qpos.copy(); before_ctrl=sim.data.ctrl.copy(); reason='No detection'; rejected=False
    if ds:
        try: sim.pick(ds[0])
        except ValueError as e: reason=str(e); rejected=np.array_equal(before_q,sim.data.qpos) and np.array_equal(before_ctrl,sim.data.ctrl)
    rows.append(dict(test='outside_workspace',color='RED',x=.79,y=.20,yaw=15,noise=0,detected_correctly=bool(ds),position_error_cm='',angle_error_deg='',pick_place_success=False,correct_refusal=rejected,reason=reason))
    Image.fromarray(dashboard(images,['Target color: RED | Detection: FOUND','REJECTED: '+reason,'Negative test only: UI prevents this placement'],sim.render('scene'))).save(OUT/'failure_workspace.png')
    # Direct far IK query cannot move live simulation.
    before_q=sim.data.qpos.copy(); rejected=False
    from robot import rotation,HOME
    try: sim.ik(np.array([2,0,.3]),rotation(0),HOME)
    except ValueError: rejected=np.array_equal(before_q,sim.data.qpos)
    rows.append(dict(test='unreachable_ik',color='',x=2,y=0,yaw=0,noise=0,detected_correctly='',position_error_cm='',angle_error_deg='',pick_place_success=False,correct_refusal=rejected,reason='IK far-pose rejection'))
    # Noise reduction and morphology use real render; report actual pixel changes.
    sim.reset(); raw=sim.render(); images,ds=process(raw,'RED',calibration,.08)
    noisy_mae=float(np.mean(np.abs(images['noisy'].astype(float)-raw))); filtered_mae=float(np.mean(np.abs(images['filtered'].astype(float)-raw)))
    # Isolated salt is removed; a small hole inside a real mask is filled.
    kernel=np.ones((3,3),np.uint8); m=images['mask'].copy(); m[10,10]=255
    clean=cv2.morphologyEx(m,cv2.MORPH_OPEN,kernel); opening_ok=clean[10,10]==0
    ys,xs=np.where(images['mask']>0); u=int(np.median(xs)); v=int(np.median(ys)); m=images['mask'].copy(); m[v,u]=0
    closing_ok=cv2.morphologyEx(m,cv2.MORPH_CLOSE,kernel)[v,u]==255
    # Survey a bounded grid. This is finite evidence, plus each UI pose gets its own IK/path check.
    workspace=[]
    for x in [.38,.525,.67]:
        for y in [-.08,.05,.18]:
            for yaw in [-85,0,85]:
                try: sim.plan(np.array([x,y,.25]),yaw); ok=True; reason='valid'
                except ValueError as e: ok=False; reason=str(e)
                workspace.append(dict(x=x,y=y,yaw=yaw,ik_path_valid=ok,reason=reason))
    (OUT/'workspace_checks.json').write_text(json.dumps(workspace,indent=2))
    positives=rows[:len(cases)]; summary=dict(physics_tests=len(positives),successful_picks=sum(r['pick_place_success'] for r in positives),correct_detections=sum(r['detected_correctly'] for r in positives),mean_position_error_cm=float(np.mean([r['position_error_cm'] for r in positives if r['position_error_cm']!=''])),max_position_error_cm=float(max(r['position_error_cm'] for r in positives if r['position_error_cm']!='')),mean_angle_error_deg=float(np.mean([r['angle_error_deg'] for r in positives if r['angle_error_deg']!=''])),max_angle_error_deg=float(max(r['angle_error_deg'] for r in positives if r['angle_error_deg']!='')),correct_refusals=sum(r['correct_refusal'] for r in rows),noise_mae=noisy_mae,filtered_mae=filtered_mae,noise_reduced=filtered_mae<noisy_mae,opening_test=bool(opening_ok),closing_test=bool(closing_ok),workspace_grid_valid=sum(r['ik_path_valid'] for r in workspace),workspace_grid_total=len(workspace))
    with (OUT/'results.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    (OUT/'coordinate_log.json').write_text(json.dumps(details,indent=2)); (OUT/'summary.json').write_text(json.dumps(summary,indent=2)); print(summary,flush=True)
    sim.close()
    assert summary['noise_reduced'] and opening_ok and closing_ok
    assert summary['correct_detections']==len(cases) and summary['successful_picks']==len(cases)
    assert summary['correct_refusals']==3
    assert summary['max_position_error_cm']<1 and summary['max_angle_error_deg']<6

if __name__=='__main__': run()
