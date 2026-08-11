import numpy as np, h5py, torch
from torch.utils.data import IterableDataset,get_worker_info
from .features import build_aux_features,build_oracle_features
from .voxel import voxelize,time_bounds

def _grid_for(det,geo,y_mode):
 if y_mode!="layerid": raise ValueError("This project requires layerid")
 if det=="ecal": return (geo.ecal_cell_size[0],1.,geo.ecal_cell_size[2]),(0.,30.),geo.ecal_xz_window
 return (geo.hcal_cell_size[0],1.,geo.hcal_cell_size[2]),(0.,48.),geo.hcal_xz_window

def _load(f,k): return f[k][:]
class MultiFileCaloIterable(IterableDataset):
 def __init__(self,h5_files,geo,time_mode,batch_size=128,split="train",train_frac=.7,shuffle_files=True,shuffle_events=True,aux_mode="energy_nhits",include_hitcount=True,include_energy_channels=True,include_propagation=False,threshold_mode="fixed_train_eventwise_median",oracle_features=False,shuffle_times=False,shard_events=False,**_):
  super().__init__(); self.geo=geo; self.time_mode=time_mode; self.batch_size=batch_size; self.split=split; self.shuffle_files=shuffle_files; self.shuffle_events=shuffle_events; self.aux_mode=aux_mode; self.include_hitcount=include_hitcount; self.include_energy_channels=include_energy_channels; self.include_propagation=include_propagation; self.threshold_mode=threshold_mode; self.oracle_features=oracle_features; self.shuffle_times=shuffle_times; self.shard_events=shard_events
  files=sorted(h5_files); cut=int(len(files)*train_frac); self.files = files if split == "all" else (files[:cut] if split == "train" else files[cut:])
 def __iter__(self):
  w=get_worker_info(); wid,wn=(0,1) if w is None else (w.id,w.num_workers); files=list(self.files if self.shard_events else self.files[wid::wn]); rng=np.random.default_rng(12345+wid)
  if self.shuffle_files:rng.shuffle(files)
  grids={d:_grid_for(d,self.geo,"layerid") for d in ("ecal","hcal")}
  for path in files:
   with h5py.File(path,"r") as f:
    D={k:_load(f,k) for k in ["trueParticleEnergy","ecal_rec_energy","ecal_rec_x","ecal_rec_z","ecal_rec_layerid","ecal_rec_time_corrected","hcal_rec_energy","hcal_rec_x","hcal_rec_z","hcal_rec_layerid","hcal_rec_time_corrected","ecal_rec_TotalEnergy","hcal_rec_TotalEnergy","ecal_rec_nhits","hcal_rec_nhits"]}
   n=len(D["trueParticleEnergy"]); idx=np.arange(n)[wid::wn] if self.shard_events else np.arange(n)
   if self.shuffle_events:rng.shuffle(idx)
   for s in range(0,n,self.batch_size):
    ids=idx[s:s+self.batch_size]; ev=[]; hv=[]; oracle=[]
    for i in ids:
     proxy=float(D["ecal_rec_TotalEnergy"][i]+D["hcal_rec_TotalEnergy"][i]); b={d:time_bounds(proxy,d,self.threshold_mode) for d in ("ecal","hcal")}
     times={d:np.asarray(D[f"{d}_rec_time_corrected"][i],np.float32).copy() for d in ("ecal","hcal")}
     if self.shuffle_times:
      for d in times: np.random.default_rng(int(i)+wid*1000003+(0 if d=="ecal" else 1)).shuffle(times[d])
     vox={}
     for d in ("ecal","hcal"):
      e=D[f"{d}_rec_energy"][i]; x=D[f"{d}_rec_x"][i]; z=D[f"{d}_rec_z"][i]; layer=D[f"{d}_rec_layerid"][i]
      vox[d]=voxelize(self.time_mode,x,layer,z,e,times[d],*grids[d],float(np.sum(e,dtype=np.float64)),bounds=b[d],include_hitcount=self.include_hitcount,include_energy_channels=self.include_energy_channels,include_propagation=self.include_propagation)
     ev.append(vox["ecal"]); hv.append(vox["hcal"])
     if self.oracle_features: oracle.append(build_oracle_features(D["ecal_rec_energy"][i],D["hcal_rec_energy"][i],times["ecal"],times["hcal"],D["ecal_rec_layerid"][i],D["hcal_rec_layerid"][i],b["ecal"],b["hcal"]))
    aux=build_aux_features(self.aux_mode,D["ecal_rec_TotalEnergy"][ids],D["hcal_rec_TotalEnergy"][ids],D["ecal_rec_nhits"][ids],D["hcal_rec_nhits"][ids],self.geo,oracle=np.stack(oracle) if oracle else None)
    yield {"ecal":torch.from_numpy(np.stack(ev)),"hcal":torch.from_numpy(np.stack(hv)),"aux":torch.from_numpy(aux),"energy_true":torch.from_numpy(D["trueParticleEnergy"][ids].astype(np.float32)).unsqueeze(1)}
