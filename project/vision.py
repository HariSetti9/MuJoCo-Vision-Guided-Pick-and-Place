"""Image-only perception. No object body positions enter this module."""
from dataclasses import dataclass
import cv2
import numpy as np

def wrap_yaw(degrees): return (float(degrees)+90)%180-90

@dataclass
class Detection:
    centre: tuple
    corners: np.ndarray
    position: np.ndarray
    yaw: float
    validity: float

class Calibration:
    def __init__(self,model,data,width=640,height=480):
        cid=model.camera('overhead').id
        self.position=data.cam_xpos[cid].copy()
        self.rotation=data.cam_xmat[cid].reshape(3,3).copy()
        self.width=width; self.height=height
        self.focal=height/(2*np.tan(np.deg2rad(model.cam_fovy[cid])/2))
        self.top_z=.27  # table .23 + uniform box height .04
    def world(self,pixel):
        u,v=pixel
        ray=self.rotation@np.array([(u-(self.width-1)/2)/self.focal,-(v-(self.height-1)/2)/self.focal,-1])
        return self.position+ray*((self.top_z-self.position[2])/ray[2])

RANGES={'BLUE':[(90,135)],'GREEN':[(35,85)],'RED':[(0,10),(170,179)],'YELLOW':[(20,34)]}

def process(rgb,color,calibration,noise=0.0,seed=42):
    noisy=rgb.copy()
    if noise:
        random=np.random.default_rng(seed).random(rgb.shape[:2]); noisy[random<noise/2]=0; noisy[(random>=noise/2)&(random<noise)]=255
    filtered=cv2.medianBlur(noisy,3)
    hsv=cv2.cvtColor(filtered,cv2.COLOR_RGB2HSV)
    mask=np.zeros(rgb.shape[:2],np.uint8)
    for low,high in RANGES[color]: mask|=cv2.inRange(hsv,np.array([low,90,50]),np.array([high,255,255]))
    raw_mask=mask.copy(); kernel=np.ones((3,3),np.uint8)
    mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel)
    mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel)
    contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    detections=[]; overlay=rgb.copy()
    for contour in contours:
        area=cv2.contourArea(contour); rect=cv2.minAreaRect(contour); w,h=rect[1]
        if min(w,h)<8 or area<150 or area>5000: continue
        aspect=max(w,h)/min(w,h); fill=area/(w*h)
        if not 1.35<aspect<3.2 or fill<.72: continue
        corners=cv2.boxPoints(rect); edges=np.roll(corners,-1,axis=0)-corners
        i=int(np.argmax(np.linalg.norm(edges,axis=1)))
        delta=calibration.world(corners[(i+1)%4])-calibration.world(corners[i])
        yaw=wrap_yaw(np.rad2deg(np.arctan2(delta[1],delta[0])))
        position=calibration.world(rect[0]); position[2]-=.02 # centre, not top
        d=Detection(tuple(rect[0]),corners,position,yaw,float(fill)); detections.append(d)
    detections.sort(key=lambda d:d.centre[0])
    for i,d in enumerate(detections):
        cv2.polylines(overlay,[d.corners.astype(np.int32)],True,(255,220,20),2)
        centre=tuple(map(int,d.centre)); cv2.circle(overlay,centre,4,(255,255,255),-1)
        cv2.putText(overlay,f'{i+1}: {color} {d.yaw:.1f} deg | valid',(centre[0]-55,centre[1]-28),cv2.FONT_HERSHEY_SIMPLEX,.42,(20,20,20),1,cv2.LINE_AA)
        cv2.putText(overlay,f'{d.position[0]:.3f}, {d.position[1]:.3f} m',(centre[0]-55,centre[1]+36),cv2.FONT_HERSHEY_SIMPLEX,.4,(20,20,20),1,cv2.LINE_AA)
    return dict(camera=rgb,noisy=noisy,filtered=filtered,raw_mask=raw_mask,mask=mask,overlay=overlay),detections

def dashboard(images,status,scene=None):
    tiles=[]
    noise_label='Simulated salt-and-pepper: ON' if np.any(images['noisy']!=images['camera']) else 'Simulated noise: OFF'
    views=[('MuJoCo scene',scene if scene is not None else images['camera']),('RGB camera',images['camera']),(noise_label,images['noisy']),('Median filter: noise reduced',images['filtered']),('Selected-color mask',images['mask']),('Detection / world estimate',images['overlay'])]
    for label,frame in views:
        if frame.ndim==2: frame=cv2.cvtColor(frame,cv2.COLOR_GRAY2RGB)
        tile=np.full((280,400,3),245,np.uint8)
        scale=min(400/frame.shape[1],250/frame.shape[0]); w=int(frame.shape[1]*scale); h=int(frame.shape[0]*scale)
        x=(400-w)//2; y=30+(250-h)//2; tile[y:y+h,x:x+w]=cv2.resize(frame,(w,h))
        cv2.putText(tile,label,(12,21),cv2.FONT_HERSHEY_SIMPLEX,.5,(25,38,55),1,cv2.LINE_AA); tiles.append(tile)
    out=np.concatenate([np.concatenate(tiles[:3],axis=1),np.concatenate(tiles[3:],axis=1)],axis=0)
    footer=np.full((80,1200,3),235,np.uint8)
    for i,line in enumerate(status[:3]): cv2.putText(footer,line,(15,22+i*23),cv2.FONT_HERSHEY_SIMPLEX,.52,(20,40,60),1,cv2.LINE_AA)
    return np.concatenate([out,footer])
