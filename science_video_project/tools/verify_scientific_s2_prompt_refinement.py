"""Verify S2-P2 checkpoints and exact Q0/Q1 reproduction of S2-P."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from pipeline.utils_io import load_metadata, load_pt
from tools.run_scientific_s0 import ATTRS, FEATURES, SEEDS, SPLIT_HASH, sha, write_json
from tools.run_scientific_s2_prompt_refinement import MODELS, forward, make_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args(); out = args.output_dir.resolve()
    config = json.loads((out / "config.json").read_text(encoding="utf-8"))
    if config["split_hash"] != SPLIT_HASH or config["features"] != FEATURES: raise ValueError("Protocol mismatch")
    metadata, feature_dir, reference = map(Path, [config["metadata"], config["feature_dir"], config["reference_checkpoint"]]); protected={str(p.resolve()):sha(p) for p in [metadata,reference]}
    data=load_metadata(metadata); data=data[data.video_id.isin({p.stem for p in feature_dir.glob("*.pt")})].copy().reset_index(drop=True); _,val_idx=train_test_split(np.arange(len(data)),test_size=.2,random_state=42,stratify=data.label)
    if (len(data),len(val_idx))!=(615,123): raise ValueError("Split mismatch")
    old=pd.read_csv(Path(config["s2p_dir"])/"interest_prompt_features.csv",dtype={"video_id":str}).set_index("video_id"); new=pd.read_csv(out/"engagement_prompt_features.csv",dtype={"video_id":str}).set_index("video_id")
    old_prompt=torch.from_numpy(old.loc[data.video_id].to_numpy(dtype=np.float32)); new_prompt=torch.from_numpy(new.loc[data.video_id].to_numpy(dtype=np.float32)); prompts={"q0_text":old_prompt,"q1_old_prompt":old_prompt,"q2_engagement_prompt":new_prompt,"q3_generic_engagement":new_prompt}
    text_rows=[]; visual_rows=[]
    for video_id in data.video_id:
        path=feature_dir/f"{video_id}.pt"; protected[str(path.resolve())]=sha(path); sample=load_pt(path); text_rows.append(np.concatenate([np.asarray(sample[k],dtype=np.float32) for k in FEATURES])); visual_rows.append(np.asarray(sample["clip_video_feat"],dtype=np.float32))
    text=torch.from_numpy(np.stack(text_rows)); visual=torch.from_numpy(np.stack(visual_rows)); saved=pd.read_csv(out/"predictions_by_model_seed.csv",dtype={"video_id":str}); rows=[]
    for kind in MODELS:
        for seed in SEEDS:
            payload=torch.load(out/"checkpoints"/f"{kind}_seed_{seed}.pt",map_location="cpu",weights_only=False); model=make_model(kind,seed,int(visual.shape[1])); model.load_state_dict(payload["model_state_dict"]); model.eval()
            with torch.inference_mode(): pred=forward(model,kind,text[val_idx],visual[val_idx],prompts[kind][val_idx]).numpy()
            expected=saved[(saved.model==kind)&(saved.seed==seed)].set_index("video_id").loc[data.iloc[val_idx].video_id][[f"pred_{a}" for a in ATTRS]].to_numpy(); error=float(np.abs(pred-expected).max())
            if error>4e-7: raise ValueError(f"Checkpoint mismatch {kind}/{seed}: {error}")
            rows.append({"model":kind,"seed":seed,"checkpoint_prediction_max_error":error,"selected_epoch":payload["selected_epoch"]})
    checks=pd.read_csv(out/"s2p_reproduction_checks.csv"); reproduction=float(checks.max_abs_error.max())
    if reproduction>1e-10: raise ValueError("Q0/Q1 reproduction failed")
    if not all(sha(p)==d for p,d in protected.items()): raise RuntimeError("Source input changed")
    result={"status":"passed","split_hash":SPLIT_HASH,"source_files":len(protected),"source_integrity":True,"s2p_reproduction_max_error":reproduction,"checkpoints":rows,"official_decision":"D"}; write_json(out/"verification.json",result); print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=="__main__": main()
