"""Panda dynamics and bounded DLS preflight. See THIRD_PARTY.md for origin."""
from pathlib import Path
import sys
import mujoco
import numpy as np
sys.path.insert(0,str(Path(__file__).parent/'vendor'))
from utils.dls_velocity_control.dls_velocity_ctrl import DLSVelocityPlanner

HOME=np.array([-.27053139,-.52082516,-.45657544,-1.98565932,-.22159819,1.51165729,.0987799])
COLORS=['blue','red','green','blue2']
BOUNDS=(.38,.67,-.08,.18)

def rotation(yaw):
    a=np.deg2rad(yaw); c,s=np.cos(a),np.sin(a)
    return np.array([[c,s,0],[s,-c,0],[0,0,-1.]])

def orient_error(target,current):
    q1=np.zeros(4); q2=np.zeros(4)
    mujoco.mju_mat2Quat(q1,target.ravel()); mujoco.mju_mat2Quat(q2,current.ravel())
    return DLSVelocityPlanner._quat_log_error(q1[[1,2,3,0]],q2[[1,2,3,0]])

class Simulation:
    def __init__(self):
        self.model=mujoco.MjModel.from_xml_path(str(Path(__file__).parent/'scene.xml'))
        self.data=mujoco.MjData(self.model); self.site=self.model.site('attachment_site').id
        self.gripper=self.model.actuator('actuator8').id
        assert self.model.actuator_trntype[self.gripper]==mujoco.mjtTrn.mjTRN_TENDON
        assert self.model.tendon('split').id==self.model.actuator_trnid[self.gripper,0]
        self.renderer=None; self.state='IDLE'; self.callback=None; self.reset()
    def reset(self):
        mujoco.mj_resetData(self.model,self.data)
        self.data.qpos[:7]=HOME; self.data.qpos[7:9]=.04
        self.data.ctrl[:7]=HOME; self.data.ctrl[self.gripper]=255
        self.state='IDLE'
        mujoco.mj_forward(self.model,self.data)
        for _ in range(250): mujoco.mj_step(self.model,self.data)
    def render(self,camera='overhead'):
        if self.renderer is None: self.renderer=mujoco.Renderer(self.model,height=480,width=640)
        if camera=='scene':
            cam=mujoco.MjvCamera(); cam.lookat[:]=[.40,0,.28]; cam.distance=1.55; cam.azimuth=135; cam.elevation=-35
        else: cam=camera
        self.renderer.update_scene(self.data,camera=cam); return self.renderer.render().copy()
    def close(self):
        if self.renderer: self.renderer.close()
    def place(self,name,x,y,yaw,validate=True):
        if self.state not in ('IDLE','DONE','REJECTED'): raise ValueError('Reset before repositioning; robot is busy')
        if not np.all(np.isfinite([x,y,yaw])): raise ValueError('Coordinates must be finite')
        if validate:
            if not self.inside([x,y]): raise ValueError('Outside validated test workspace')
            if abs(yaw)>85: raise ValueError('Yaw must be between -85 and +85 degrees')
            for other in COLORS:
                if other!=name and np.linalg.norm(self.data.xpos[self.model.body(other).id,:2]-[x,y])<.095: raise ValueError('Boxes too close: keep 9.5 cm centre separation')
            self.plan(np.array([x,y,.25]),yaw)
        joint=self.model.joint(name+'_free'); adr=joint.qposadr[0]; vel=joint.dofadr[0]
        self.data.qpos[adr:adr+7]=[x,y,.25,np.cos(np.deg2rad(yaw)/2),0,0,np.sin(np.deg2rad(yaw)/2)]
        self.data.qvel[vel:vel+6]=0; mujoco.mj_forward(self.model,self.data)
    @staticmethod
    def inside(p): return BOUNDS[0]<=p[0]<=BOUNDS[1] and BOUNDS[2]<=p[1]<=BOUNDS[3]
    def ik(self,pos,R,initial):
        scratch=mujoco.MjData(self.model); scratch.qpos[:]=self.data.qpos; scratch.qpos[:7]=initial
        jacp=np.zeros((3,self.model.nv)); jacr=jacp.copy()
        for _ in range(350):
            mujoco.mj_forward(self.model,scratch)
            dp=pos-scratch.site_xpos[self.site]; dr=orient_error(R,scratch.site_xmat[self.site].reshape(3,3))
            if np.linalg.norm(dp)<.001 and np.linalg.norm(dr)<.015: return scratch.qpos[:7].copy()
            mujoco.mj_jacSite(self.model,scratch,jacp,jacr,self.site)
            J=np.vstack([jacp[:,:7],jacr[:,:7]])
            # Same damped least-squares method as the attributed workshop planner.
            dq=J.T@np.linalg.solve(J@J.T+.0001*np.eye(6),np.r_[dp,dr])
            dq*=min(1,.12/max(np.max(np.abs(dq)),1e-9))
            scratch.qpos[:7]=np.clip(scratch.qpos[:7]+dq,self.model.jnt_range[:7,0]+.01,self.model.jnt_range[:7,1]-.01)
        raise ValueError('IK cannot reach position and gripper angle')
    def plan(self,position,yaw):
        if not self.inside(position): raise ValueError('Outside validated test workspace')
        if not .235<=position[2]<=.265: raise ValueError('Estimated box height is invalid')
        R=rotation(yaw); x,y,z=position
        waypoints=[('APPROACHING',[x,y,z+.16],255),('DESCENDING',[x,y,z+.015],255),('GRASPING',[x,y,z+.015],0),('LIFTING',[x,y,z+.18],0),('TRANSFERRING',[.47,-.245,.45],0),('PLACING',[.47,-.245,.28],0),('RELEASING',[.47,-.245,.28],255),('RETREATING',[.47,-.245,.46],255)]
        q=self.data.qpos[:7].copy(); plan=[]
        for label,pos,grip in waypoints:
            q=self.ik(np.array(pos),R,q); plan.append((label,q.copy(),grip))
        # Verify sampled joint paths on scratch data before allowing motion.
        # Boxes are excluded here; separation and vertical approach are checked separately.
        scratch=mujoco.MjData(self.model); start=self.data.qpos[:7].copy()
        for label,q,grip in plan:
            for fraction in np.linspace(0,1,15):
                scratch.qpos[:]=self.data.qpos; scratch.qpos[:7]=start+(q-start)*fraction
                mujoco.mj_forward(self.model,scratch)
                for contact in scratch.contact:
                    if contact.dist>=-.003: continue
                    names=[self.model.geom(g).name for g in (contact.geom1,contact.geom2)]
                    bodies=[int(self.model.geom_bodyid[g]) for g in (contact.geom1,contact.geom2)]
                    if 'table' in names and any(self.model.body(b).name in ('hand','left_finger','right_finger','link5','link6','link7') for b in bodies):
                        raise ValueError('Preflight predicts robot/table collision')
            start=q
        return plan
    def pick(self,detection):
        try: plan=self.plan(detection.position,detection.yaw)
        except ValueError:
            self.state='REJECTED'; raise
        # Nothing writes object qpos after this point. All transport is contact physics.
        for label,q,grip in plan:
            self.state=label; start=self.data.ctrl[:7].copy()
            duration=1.0 if label in ('GRASPING','RELEASING') else max(1.5,float(np.max(np.abs(q-start)))/.45)
            steps=int(duration/self.model.opt.timestep)
            for k in range(steps+250):
                t=min(1,k/steps); blend=t*t*(3-2*t)
                self.data.ctrl[:7]=start+(q-start)*blend; self.data.ctrl[self.gripper]=grip
                mujoco.mj_step(self.model,self.data)
                if self.callback and k%25==0: self.callback()
            error=np.linalg.norm(self.data.site_xpos[self.site]-self._site_at(q))
            if error>.025: self.state='REJECTED'; raise ValueError(f'Controller tracking error {error:.3f} m')
        start=self.data.ctrl[:7].copy(); self.state='RETURNING'
        for k in range(1800):
            t=min(1,k/1500); self.data.ctrl[:7]=start+(HOME-start)*(t*t*(3-2*t)); mujoco.mj_step(self.model,self.data)
            if self.callback and k%25==0: self.callback()
        self.state='DONE'
    def _site_at(self,q):
        scratch=mujoco.MjData(self.model); scratch.qpos[:]=self.data.qpos; scratch.qpos[:7]=q; mujoco.mj_forward(self.model,scratch)
        return scratch.site_xpos[self.site].copy()
