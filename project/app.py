"""One classroom perception dashboard plus optional native MuJoCo viewer."""
import argparse,time,json,subprocess,sys
from pathlib import Path
import tkinter as tk
from tkinter import ttk
import cv2,numpy as np
from PIL import Image,ImageTk
import imageio.v2 as imageio
from robot import Simulation,COLORS,BOUNDS
from vision import Calibration,process,dashboard

STAGES=['Camera image captured','Noise added or disabled','Image filtered: noise reduced','Selected-color mask created','Box outline, centre and angle detected','World position estimated','Reachability checked','Robot picks and places the box']

class App:
    def __init__(self,smoke=False):
        self.root=tk.Tk(); self.root.title('MuJoCo Vision-Guided Multi-Object Pick-and-Place'+(' - Validation' if smoke else ''))
        def report_error(kind,value,tb):
            import traceback
            Path(__file__).parent.joinpath('outputs/gui_errors.log').write_text(''.join(traceback.format_exception(kind,value,tb)))
        self.root.report_callback_exception=report_error
        self.root.geometry('1260x900'); self.root.configure(bg='#f2f4f6')
        self.sim=Simulation(); self.cal=Calibration(self.sim.model,self.sim.data)
        self.viewer=None; self.writer=None; self.busy=False; self.paused=False; self.closed=False
        self.stage=0; self.frozen=False; self.selected=None; self.images={}; self.ds=[]; self.photos=[]; self.smoke=smoke
        self.color=tk.StringVar(value='BLUE'); self.noise=tk.BooleanVar(value=False); self.step_mode=tk.BooleanVar(value=True); self.record=tk.BooleanVar(value=True)
        self.box=tk.StringVar(value='blue'); self.x=tk.StringVar(value='.50'); self.y=tk.StringVar(value='.03'); self.yaw=tk.StringVar(value='0')
        self.status=tk.StringVar(value='READY - Choose a color; two blue boxes require a click')
        self.info=tk.StringVar(value='Workspace: x 0.38-0.67 m | y -0.08-0.18 m | yaw -85 to +85 deg. Each pose is checked by IK.')
        top=ttk.Frame(self.root,padding=10); top.pack(fill='x')
        ttk.Label(top,text='Target color:').pack(side='left')
        self.combo=ttk.Combobox(top,textvariable=self.color,values=['BLUE','RED','GREEN','YELLOW'],width=9,state='readonly'); self.combo.pack(side='left',padx=6); self.combo.bind('<<ComboboxSelected>>',lambda e:self.invalidate())
        ttk.Checkbutton(top,text='Simulated noise (8%)',variable=self.noise,command=self.invalidate).pack(side='left',padx=8)
        ttk.Checkbutton(top,text='Step-through',variable=self.step_mode).pack(side='left',padx=8)
        ttk.Checkbutton(top,text='Record MP4',variable=self.record).pack(side='left',padx=8)
        self.buttons=[]
        for label,command in [('Detect / start stages',self.detect),('Next stage',self.next_stage),('Run pick-and-place',self.run_pick),('Reset',self.reset),('MuJoCo viewer',self.open_viewer)]:
            b=ttk.Button(top,text=label,command=command); b.pack(side='left',padx=3); self.buttons.append(b)
        controls=ttk.Frame(self.root,padding=10); controls.pack(fill='x')
        ttk.Label(controls,text='Place box:').pack(side='left'); ttk.Combobox(controls,textvariable=self.box,values=COLORS,width=8,state='readonly').pack(side='left',padx=5)
        for label,var in [('X (m)',self.x),('Y (m)',self.y),('Yaw (deg)',self.yaw)]:
            ttk.Label(controls,text=label).pack(side='left',padx=5); ttk.Entry(controls,textvariable=var,width=8).pack(side='left')
        b=ttk.Button(controls,text='Validate and place',command=self.place); b.pack(side='left',padx=10); self.buttons.append(b)
        b=ttk.Button(controls,text='Run measured test mode',command=self.test_mode); b.pack(side='left',padx=5); self.buttons.append(b)
        ttk.Button(controls,text='Pause / resume robot',command=self.pause).pack(side='left',padx=10)
        ttk.Label(self.root,textvariable=self.info,padding=8).pack(fill='x')
        ttk.Label(self.root,textvariable=self.status,font=('Segoe UI',13,'bold'),padding=8,wraplength=1200).pack(fill='x')
        self.stage_label=ttk.Label(self.root,text='Step 0 / 8: ready',padding=5); self.stage_label.pack(fill='x')
        grid=ttk.Frame(self.root); grid.pack(expand=True)
        self.views=[]
        for i,label in enumerate(['RGB camera - click a matching box','Noisy image - simulated','Median-filtered image - noise reduced','Target mask - selected color only','Detection overlay','MuJoCo scene']):
            panel=ttk.Frame(grid,padding=4); panel.grid(row=i//3,column=i%3)
            ttk.Label(panel,text=label,font=('Segoe UI',10,'bold')).pack()
            view=tk.Label(panel,bg='#dce2e7'); view.pack(); self.views.append(view)
        self.views[0].bind('<Button-1>',self.click)
        ttk.Label(self.root,text='Filter reduces image noise | Segmentation separates color | Detection selects a region | Localization estimates pose | IK checks reachability',padding=10).pack(fill='x')
        self.root.protocol('WM_DELETE_WINDOW',self.close); self.sim.callback=self.robot_frame
        self.tick()
        if smoke: self.root.after(1000,self.smoke_test)
    def invalidate(self):
        if self.busy: return
        self.frozen=False; self.stage=0; self.selected=None
        self.stage_label.configure(text='Step 0 / 8: ready')
        self.status.set('Preview updated. Click Detect / start stages to capture a target.')
    def refresh(self):
        if not self.frozen:
            self.images,self.ds=process(self.sim.render(),self.color.get(),self.cal,.08 if self.noise.get() else 0)
            self.selected=0 if len(self.ds)==1 else None
        self.photos=[]
        scene=self.sim.render('scene')
        order=['camera','noisy','filtered','mask','overlay']
        for i,key in enumerate(order):
            show=self.images[key].copy()
            if key=='overlay' and self.selected is not None and self.selected<len(self.ds):
                cv2.polylines(show,[self.ds[self.selected].corners.astype(np.int32)],True,(255,80,0),3)
            # Reveal stages in order when paused; blank panels are deliberately labeled.
            required=[1,2,3,4,5][i]
            if self.frozen and self.step_mode.get() and self.stage<required:
                show=np.full((480,640,3),230,np.uint8)
                cv2.putText(show,'Press Next stage to reveal',(75,240),cv2.FONT_HERSHEY_SIMPLEX,.7,(45,55,65),1)
            if show.ndim==2: show=cv2.cvtColor(show,cv2.COLOR_GRAY2RGB)
            photo=ImageTk.PhotoImage(Image.fromarray(show).resize((400,300))); self.photos.append(photo); self.views[i].configure(image=photo)
        photo=ImageTk.PhotoImage(Image.fromarray(scene).resize((400,300))); self.photos.append(photo); self.views[5].configure(image=photo)
        if self.viewer and self.viewer.is_running(): self.viewer.sync()
        return scene
    def tick(self):
        if self.closed: return
        if not self.busy:
            self.refresh()
        self.root.after(180,self.tick)
    def detect(self):
        if self.busy: return
        self.frozen=False; self.refresh(); self.frozen=True; self.stage=1
        self.status.set('Camera captured. Press Next stage.' if self.step_mode.get() else 'Detection complete')
        if not self.step_mode.get(): self.stage=6
        self.show_stage()
    def show_stage(self):
        self.stage_label.configure(text=f'Step {self.stage} / 8: {STAGES[self.stage-1]}' if self.stage else 'Step 0 / 8: ready')
        if self.stage>=5:
            if not self.ds: self.status.set(f'Target color: {self.color.get()} | Detection: NOT FOUND | REJECTED: no valid target; robot has not moved')
            elif self.selected is None: self.status.set(f'Detection: {len(self.ds)} matches. Click the specific box in RGB camera preview.')
            else:
                d=self.ds[self.selected]; msg=f'Target color: {self.color.get()} | Detection: FOUND | Shape validity: {d.validity:.2f} | Image centre: ({d.centre[0]:.1f}, {d.centre[1]:.1f}) | Yaw: {d.yaw:.1f} deg'
                if self.stage>=6: msg+=f' | World: ({d.position[0]:.3f}, {d.position[1]:.3f}, {d.position[2]:.3f}) m'
                self.status.set(msg)
        self.refresh()
    def click(self,event):
        if self.busy or not self.frozen or self.stage<5: return
        p=(event.x*640/400,event.y*480/300)
        for i,d in enumerate(self.ds):
            if cv2.pointPolygonTest(d.corners.astype(np.float32),p,False)>=0:
                self.selected=i; self.show_stage(); break
    def next_stage(self):
        if self.busy: return
        if not self.frozen: self.detect(); return
        if self.stage>=6:
            if self.selected is None: self.status.set('REJECTED: choose one valid detected target before continuing'); return
            try: self.sim.plan(self.ds[self.selected].position,self.ds[self.selected].yaw)
            except ValueError as e: self.status.set('REJECTED: '+str(e)); return
        if self.stage==7: self.stage=8; self.run_pick(); return
        self.stage=min(7,self.stage+1); self.show_stage()
        if self.stage==7: self.status.set(self.status.get()+' | Reachability: PASSED; robot has not moved')
    def run_pick(self):
        if self.busy: return
        if not self.frozen: self.detect()
        if self.step_mode.get() and self.stage<7: self.status.set('Use Next stage through reachability before starting the robot'); return
        if self.selected is None: self.status.set('REJECTED: missing or ambiguous target; click one matching box'); return
        d=self.ds[self.selected]
        try: self.sim.plan(d.position,d.yaw)
        except ValueError as e: self.status.set('REJECTED: '+str(e)); return
        self.busy=True; self.frozen=False; self.stage=8; self.paused=False; self.locked=d
        self.stage_label.configure(text='Step 8 / 8: Robot picks and places the box')
        for b in self.buttons: b.configure(state='disabled')
        out=Path(__file__).parent/'outputs'; out.mkdir(exist_ok=True)
        if self.record.get(): self.writer=imageio.get_writer(out/'live_demo.mp4',fps=20,codec='libx264',quality=7,macro_block_size=16)
        try:
            self.sim.pick(d); self.status.set('DONE - sequence finished. Visually confirm box in tray; measured success is reported by test mode.')
        except (ValueError,RuntimeError) as e: self.status.set('STOPPED: '+str(e))
        finally:
            if self.writer: self.writer.close(); self.writer=None
            self.busy=False; self.frozen=True
            if not self.closed:
                for b in self.buttons: b.configure(state='normal')
    def robot_frame(self):
        if self.closed: raise RuntimeError('Window closed; motion stopped')
        while self.paused:
            self.root.update(); time.sleep(.03)
            if self.closed: raise RuntimeError('Window closed; motion stopped')
        # Locked target estimates stay visible while live masks may change under occlusion.
        self.images,self.ds=process(self.sim.render(),self.color.get(),self.cal,.08 if self.noise.get() else 0)
        self.frozen=True; scene=self.refresh(); self.frozen=False
        d=self.locked
        self.status.set(f'Target color: {self.color.get()} | Locked image position: {d.position.round(3)} m | Yaw: {d.yaw:.1f} deg | Robot: {self.sim.state}')
        self.root.update()
        if self.writer: self.writer.append_data(dashboard(self.images,[self.status.get(),'All transport through gripper contact physics','Simulated noise is separate from distractor boxes'],scene))
    def pause(self):
        if self.busy: self.paused=not self.paused; self.status.set('PAUSED - robot physics frozen' if self.paused else 'RESUMING')
    def place(self):
        if self.busy: return
        try:
            self.sim.place(self.box.get(),float(self.x.get()),float(self.y.get()),float(self.yaw.get())); self.invalidate(); self.status.set('Placement accepted: bounds, separation, IK and sampled table-clearance checks passed')
        except ValueError as e: self.status.set('REJECTED: '+str(e))
    def reset(self):
        if self.busy: return
        self.sim.reset(); self.invalidate(); self.status.set('Scene reset. Choose color and detect.')
    def test_mode(self):
        if self.busy: return
        if hasattr(self,'test_process') and self.test_process.poll() is None:
            self.status.set('Test mode is already running'); return
        self.test_log=open(Path(__file__).parent/'outputs/test_mode.log','w')
        self.test_process=subprocess.Popen([sys.executable,str(Path(__file__).parent/'run_tests.py')],stdout=self.test_log,stderr=self.test_log)
        self.status.set('Measured test mode running in a separate simulation; CSV and outputs will be saved.')
        self.root.after(1000,self.poll_tests)
    def poll_tests(self):
        if self.closed: return
        code=self.test_process.poll()
        if code is None: self.root.after(1000,self.poll_tests); return
        self.test_log.close()
        if code==0:
            result=json.loads((Path(__file__).parent/'outputs/summary.json').read_text())
            self.status.set(f'TESTS PASSED: {result["successful_picks"]}/{result["physics_tests"]} picks; mean position error {result["mean_position_error_cm"]:.3f} cm; CSV saved.')
        else: self.status.set('Test mode failed: inspect outputs/test_mode.log and results.csv')
    def open_viewer(self):
        if self.viewer and self.viewer.is_running(): return
        import mujoco.viewer
        self.viewer=mujoco.viewer.launch_passive(self.sim.model,self.sim.data)
        self.viewer.cam.lookat[:]=[.4,0,.28]; self.viewer.cam.distance=1.55; self.viewer.cam.azimuth=135; self.viewer.cam.elevation=-35
    def smoke_test(self):
        # Exercise the actual callbacks used by the buttons and preview.
        self.color.set('BLUE'); self.detect()
        for _ in range(4): self.next_stage()
        assert len(self.ds)==2 and self.selected is None
        from types import SimpleNamespace
        centre=self.ds[1].centre
        self.click(SimpleNamespace(x=centre[0]*400/640,y=centre[1]*300/480))
        assert self.selected==1
        multiple_selection=True
        self.reset()
        self.noise.set(True); self.color.set('RED'); self.detect()
        for _ in range(6): self.next_stage()
        self.record.set(False); self.run_pick()
        red=self.sim.data.xpos[self.sim.model.body('red').id].copy()
        red_in_tray=abs(red[0]-.47)<.073 and abs(red[1]+.245)<.053
        self.color.set('YELLOW'); self.invalidate(); self.detect()
        for _ in range(4): self.next_stage()
        assert not self.ds and 'REJECTED' in self.status.get()
        Path(__file__).parent.joinpath('outputs/gui_smoke.json').write_text(json.dumps(dict(stage=self.stage,status=self.status.get(),robot_state=self.sim.state,window_open=True,multiple_instance_click_passed=multiple_selection,red_physical_placement=bool(red_in_tray)),indent=2))
    def close(self):
        self.closed=True
        if self.viewer: self.viewer.close()
        if not self.busy: self.sim.close()
        self.root.destroy()
    def run(self): self.root.mainloop()

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--gui-smoke',action='store_true'); args=parser.parse_args()
    App(args.gui_smoke).run()
