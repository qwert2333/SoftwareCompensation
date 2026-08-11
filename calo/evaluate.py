import json,os
import h5py,numpy as np,torch
from torch.utils.data import DataLoader
from .dataset import MultiFileCaloIterable
from .features import aux_dim
from .geo import build_geo
from .model import TimeAwareFEMNet
from .voxel import channels_for
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE="/gpfs/workdir/xiax/Data_V0/Test_Samples/fixed_direction"
PARTICLES={
 "pi+":("test_samples_pi+.h5",0.13957039),
 "K-":("test_samples_K-.h5",0.493677),
 "KL0":("test_samples_KL0.h5",0.497611),
 "n":("test_samples_n.h5",0.9395654205),
 "e-":("test_samples_e-.h5",0.00051099895),
}
NOMINAL_MOMENTA=np.array([1.,2.,3.,4.,5.,6.,7.,8.,9.,10.,11.,12.,13.,14.,15.,20.,30.,40.,50.,60.,80.,100.,120.,150.,170.,190.])
def nominal_energies(mass): return np.sqrt(NOMINAL_MOMENTA**2+mass**2)
def map_true_energy(values,mass):
 grid=nominal_energies(mass); v=np.asarray(values,np.float64); return grid[np.argmin(abs(v[:,None]-grid[None,:]),axis=1)]
def validate_groups(values,mass,expected=15000):
 mapped=map_true_energy(values,mass); grid=nominal_energies(mass); counts={float(e):int(np.sum(mapped==e)) for e in grid}; bad={e:n for e,n in counts.items() if n!=expected}
 if bad: raise RuntimeError(f"Expected {expected} events/nominal point, got {bad}")
 return mapped,counts
def evaluate_one(exp,out_dir):
 geo=build_geo(**exp["geo"]); mode=exp["time_mode"]; hit=bool(exp.get("include_hitcount",False)); energy=bool(exp.get("include_energy_channels",True)); prop=bool(exp.get("include_propagation",False)); oracle=bool(exp.get("oracle_features",False)); aux_mode=exp.get("aux_mode","energy_nhits")
 model=TimeAwareFEMNet(channels_for(mode,hit,energy,prop),aux_dim(aux_mode,oracle),bool(exp.get("use_longitudinal_profile",False))).to(DEVICE)
 ckpt=torch.load(os.path.join(out_dir,"checkpoints","best.pth"),map_location=DEVICE); model.load_state_dict(ckpt["model"]); model.eval()
 combined={}
 for tag,(filename,mass) in PARTICLES.items():
  path=os.path.join(BASE,filename)
  with h5py.File(path,"r") as f: validate_groups(f["trueParticleEnergy"][:],mass)
  ds=MultiFileCaloIterable([path],geo,mode,batch_size=128,split="all",shuffle_files=False,shuffle_events=False,aux_mode=aux_mode,include_hitcount=hit,include_energy_channels=energy,include_propagation=prop,threshold_mode=exp.get("threshold_mode","fixed_train_eventwise_median"),oracle_features=oracle,shuffle_times=bool(exp.get("shuffle_times",False)),shard_events=True)
  preds=[]; truths=[]
  with torch.no_grad():
   for b in DataLoader(ds,batch_size=None,num_workers=1,pin_memory=DEVICE.type=="cuda",persistent_workers=True,prefetch_factor=2):
    preds.append(model(b["ecal"].to(DEVICE,non_blocking=True),b["hcal"].to(DEVICE,non_blocking=True),b["aux"].to(DEVICE,non_blocking=True)).cpu().numpy()); truths.append(b["energy_true"].squeeze(1).numpy())
  pred=np.concatenate(preds); raw=np.concatenate(truths); mapped,counts=validate_groups(raw,mass); pdir=os.path.join(out_dir,"prediction_results",tag); os.makedirs(pdir,exist_ok=True)
  with h5py.File(os.path.join(pdir,"predictions.h5"),"w") as f: f["true_energy_raw"]=raw; f["nominal_energy"]=mapped; f["pred_energy"]=pred
  summary={}
  for p,e in zip(NOMINAL_MOMENTA,nominal_energies(mass)):
   x=pred[mapped==e]; mu=float(np.mean(x)); sig=float(np.std(x)); summary[f"{p:g}"]={"nominal_momentum_GeV":float(p),"nominal_energy_GeV":float(e),"N":len(x),"mean":mu,"sigma":sig,"resolution":sig/max(abs(mu),1e-9),"relative_bias":(mu-float(e))/float(e)}
  with open(os.path.join(pdir,"summary.json"),"w") as f: json.dump(summary,f,indent=2)
  with open(os.path.join(pdir,"group_counts.json"),"w") as f: json.dump({str(k):v for k,v in counts.items()},f,indent=2)
  combined[tag]=summary; print(f"[EVAL] {tag}: {len(pred)} events, 26 x 15,000 validated")
 with open(os.path.join(out_dir,"prediction_results","five_particle_summary.json"),"w") as f: json.dump(combined,f,indent=2)
