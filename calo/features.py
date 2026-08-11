import numpy as np

N_LAYERS = {"ecal": 30, "hcal": 48}
ORACLE_DIM = 11 + 5 * (30 + 48)

def _weighted_stats(e, t):
    e = np.asarray(e, np.float64); t = np.asarray(t, np.float64)
    good = np.isfinite(e) & np.isfinite(t) & (e > 0)
    if not np.any(good): return 0.0, 0.0, np.zeros(3, np.float32)
    e, t = e[good], t[good]; w = e / e.sum()
    mean = float(np.sum(w*t)); rms = float(np.sqrt(max(np.sum(w*(t-mean)**2), 0.0)))
    return mean, rms, np.quantile(t, [0.1, 0.5, 0.9]).astype(np.float32)

def _layer_timing(e, t, layer, n_layers, early, late):
    out = np.zeros((5, n_layers), np.float32)
    e=np.asarray(e,np.float64); t=np.asarray(t,np.float64); layer=np.asarray(layer,np.int64)
    for lid in range(n_layers):
        m=(layer==lid)&np.isfinite(t)&np.isfinite(e)&(e>0)
        if not np.any(m): continue
        ee,tt=e[m],t[m]; total=ee.sum()
        out[0,lid]=ee[tt<early].sum()/total
        out[1,lid]=ee[(tt>=early)&(tt<late)].sum()/total
        out[2,lid]=ee[tt>=late].sum()/total
        out[3,lid]=np.min(tt)
        mu=np.sum(ee*tt)/total
        out[4,lid]=np.sqrt(max(np.sum(ee*(tt-mu)**2)/total,0.0))
    return out.reshape(-1)

def build_oracle_features(ecal_e,hcal_e,ecal_t,hcal_t,ecal_layer,hcal_layer,ecal_bounds,hcal_bounds):
    em,er,eq=_weighted_stats(ecal_e,ecal_t); hm,hr,hq=_weighted_stats(hcal_e,hcal_t)
    he=np.asarray(hcal_e,np.float64); ht=np.asarray(hcal_t,np.float64)
    valid=np.isfinite(he)&np.isfinite(ht)&(he>0); total=he[valid].sum()
    hlate=float(he[valid&(ht>=hcal_bounds[-1])].sum()/total) if total>0 else 0.0
    scalars=np.array([em,er,*eq,hm,hr,*hq,hlate],np.float32)
    el=_layer_timing(ecal_e,ecal_t,ecal_layer,30,ecal_bounds[0],ecal_bounds[-1])
    hl=_layer_timing(hcal_e,hcal_t,hcal_layer,48,hcal_bounds[0],hcal_bounds[-1])
    v=np.concatenate([scalars,el,hl]).astype(np.float32)
    return np.sign(v)*np.log1p(np.abs(v))

def build_aux_features(aux_mode,energy_ecal,energy_hcal,ecal_nhits,hcal_nhits,geo,oracle=None):
    n=len(np.asarray(energy_ecal))
    if aux_mode=="none": base=np.zeros((n,0),np.float32)
    else:
        cols=[np.log1p(np.asarray(energy_ecal,np.float32)),np.log1p(np.asarray(energy_hcal,np.float32))]
        if aux_mode=="energy_nhits": cols += [np.log1p(np.asarray(ecal_nhits,np.float32)),np.log1p(np.asarray(hcal_nhits,np.float32))]
        elif aux_mode!="energy": raise ValueError(f"Unsupported aux_mode: {aux_mode}")
        base=np.stack(cols,axis=1).astype(np.float32)
    if oracle is not None: base=np.concatenate([base,np.asarray(oracle,np.float32).reshape(n,-1)],axis=1)
    return base

def aux_dim(aux_mode,oracle_features=False):
    return {"none":0,"energy":2,"energy_nhits":4}[aux_mode] + (ORACLE_DIM if oracle_features else 0)
