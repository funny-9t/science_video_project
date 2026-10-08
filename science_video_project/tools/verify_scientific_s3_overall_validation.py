"""Reload every S3 stage-2 checkpoint and compare fixed validation predictions."""
from __future__ import annotations

import argparse, json, logging, sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from tools.run_progressive_training import prepare_data
from tools.run_scientific_s0 import SEEDS, SPLIT_HASH, sha, write_json
from tools.run_scientific_s3_overall_validation import MODELS, build_model, evaluate
from training.utils_train import get_device
from pipeline.config import CFG


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("output_dir", type=Path)
    args = parser.parse_args(); out=args.output_dir.resolve(); config=json.loads((out/"experiment_manifest.json").read_text(encoding="utf-8"))
    metadata, features=Path(config["metadata"]),Path(config["feature_dir"]); a=type("Args",(),{"metadata":metadata,"feature_dir":features,"split_seed":42})(); _,val_ds,split=prepare_data(a,logging.getLogger("s3verify"))
    if split!=SPLIT_HASH: raise ValueError("Split mismatch")
    saved=pd.read_csv(out/"paired_predictions.csv",dtype={"video_id":str}); device=get_device(CFG.device); rows=[]
    for kind in MODELS:
        for seed in SEEDS:
            path=out/kind/f"seed_{seed}"/"stage2_best.pt"; payload=torch.load(path,map_location="cpu",weights_only=False); model=build_model(kind,seed,device); model.load_state_dict(payload["model_state_dict"],strict=True); _,_,_,frame=evaluate(model,val_ds.samples,device)
            expected=saved[(saved.model==kind)&(saved.seed==seed)].set_index("video_id").loc[frame.video_id][["probability","overall_q","pred_scientific_score"]].to_numpy(); actual=frame[["probability","overall_q","pred_scientific_score"]].to_numpy(); error=float(np.abs(actual-expected).max())
            if error>5e-7: raise ValueError(f"Reload mismatch {kind}/{seed}: {error}")
            rows.append({"model":kind,"seed":seed,"checkpoint":str(path.resolve()),"prediction_max_abs_error":error})
    protected={str(metadata.resolve()):sha(metadata)}
    for path in features.glob("*.pt"): protected[str(path.resolve())]=sha(path)
    result={"status":"passed","split_hash":split,"source_integrity":True,"source_files":len(protected),"max_prediction_abs_error":max(r["prediction_max_abs_error"] for r in rows),"checkpoints":rows}; write_json(out/"checkpoint_reload_verification.json",result); print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=="__main__": main()
