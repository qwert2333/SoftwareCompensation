import csv
import os
import numpy as np
EPS=1e-6; C_MM_PER_NS=299.792458
TIME_INFO="/gpfs/users/xiax/SoftwareCompensation/pytorch/projects_0/time_info"; QUANTILES=("15%","30%","50%","80%")
def _read(filename,edep):
 r={"ecal":{},"hcal":{}}
 with open(os.path.join(TIME_INFO,filename),newline="") as f:
  for row in csv.DictReader(f):
   d=row["detector"]
   if d not in r or row["quantile_label"] not in QUANTILES: continue
   k=float(row["momentum_GeV_c"]) if edep else "common"
   r[d].setdefault(k,{})[row["quantile_label"]]=float(row["eventwise_median"])
 return {d:{k:tuple(v[q] for q in QUANTILES) for k,v in g.items()} for d,g in r.items()}
FIXED=_read("trainset_quantile_mean_median_comparison.csv",False)
EDEP=_read("ipstart_isotropic_eval_pi+_quantile_mean_median_comparison.csv",True)
def time_bounds(E,det,mode):
 if mode=="fixed_train_eventwise_median": return FIXED[det]["common"]
 if mode=="energy_dependent_eval_eventwise_median":
  grid=np.asarray(sorted(EDEP[det])); return EDEP[det][float(grid[np.argmin(abs(grid-float(E)))])]
 raise ValueError(mode)
def channels_for(mode,include_hitcount=False,include_energy_channels=True,include_propagation=False):
 return (2 if include_energy_channels else 0)+(1 if include_hitcount else 0)+(5 if mode=="bins5" else 0)+(1 if include_propagation else 0)
def voxelize(mode,x,y,z,e,t,cell_size,y_range,xz_window,E_total_full,bounds=None,include_hitcount=False,include_energy_channels=True,include_propagation=False):
 x=np.asarray(x,np.float32); y=np.asarray(y,np.float32); z=np.asarray(z,np.float32); e=np.asarray(e,np.float32); t=np.asarray(t,np.float32)
 nx=int(xz_window[0]/cell_size[0]); ny=int((y_range[1]-y_range[0])/cell_size[1]); nz=int(xz_window[1]/cell_size[2]); C=channels_for(mode,include_hitcount,include_energy_channels,include_propagation)
 if len(e)==0 or e.sum()<=0:return np.zeros((C,nx,ny,nz),np.float32)
 xc=np.sum(x*e)/e.sum(); zc=np.sum(z*e)/e.sum(); xmin=xc-xz_window[0]/2; zmin=zc-xz_window[1]/2
 m=(x>=xmin)&(x<xmin+xz_window[0])&(y>=y_range[0])&(y<y_range[1])&(z>=zmin)&(z<zmin+xz_window[1]); x,y,z,e,t=x[m],y[m],z[m],e[m],t[m]
 if len(e)==0:return np.zeros((C,nx,ny,nz),np.float32)
 ix=np.clip(((x-xmin)/cell_size[0]).astype(int),0,nx-1); iy=np.clip(((y-y_range[0])/cell_size[1]).astype(int),0,ny-1); iz=np.clip(((z-zmin)/cell_size[2]).astype(int),0,nz-1); flat=ix*ny*nz+iy*nz+iz; n=nx*ny*nz
 ve=np.bincount(flat,weights=e,minlength=n).astype(np.float32).reshape(nx,ny,nz); out=[]
 if include_energy_channels:
  out += [ve/max(float(ve.max()),EPS),ve/(float(E_total_full)+EPS)]
 if include_hitcount:
  h=np.bincount(flat,minlength=n).astype(np.float32).reshape(nx,ny,nz); out.append(h/max(float(h.max()),1.0))
 if mode=="bins5":
  if bounds is None or len(bounds)!=4:raise ValueError("bins5 needs 4 bounds")
  nzmask=ve>0
  for lo,hi in zip((-np.inf,*bounds),(*bounds,np.inf)):
   eb=np.bincount(flat,weights=e*((t>=lo)&(t<hi)),minlength=n).astype(np.float32).reshape(nx,ny,nz); f=np.zeros_like(ve); f[nzmask]=eb[nzmask]/ve[nzmask]; out.append(f)
 if include_propagation:
  ptime=np.sqrt(x*x+y*y+z*z)/C_MM_PER_NS; ep=np.bincount(flat,weights=e*ptime,minlength=n).astype(np.float32).reshape(nx,ny,nz); p=np.zeros_like(ve); nzmask=ve>0; p[nzmask]=ep[nzmask]/ve[nzmask]; out.append(p/10.0)
 return np.stack(out).astype(np.float32)
