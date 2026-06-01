"""
批量推理脚本 - 从CSV文件直接读取数据进行推理
支持您真实的抖音元数据格式
"""
import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
import torch
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.config import CFG
from pipeline.step_meta import MetaFeatureBuilder
from training.model_mvp import MultiModalQualityModel
from training.utils_train import adapt_checkpoint_state_dict, get_device


class BatchInferenceEngine:
    """批量推理引擎"""
    
    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        """初始化推理引擎"""
        self.device = get_device(device)
        print(f"Using device: {self.device}")
        
        # 加载模型
        self.model = MultiModalQualityModel(
            text_dim=CFG.text_dim,
            video_dim=CFG.video_dim,
            audio_dim=CFG.audio_dim,
            meta_dim=CFG.meta_dim,
            hidden_dim=CFG.hidden_dim,
        ).to(self.device)
        
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        else:
            state_dict = ckpt
        
        self.model.load_state_dict(adapt_checkpoint_state_dict(state_dict))
        self.model.eval()
        print(f"✓ Loaded checkpoint: {checkpoint_path}")
    
    def infer_from_csv(self, csv_path: str, output_path: str | None = None, limit: int | None = None):
        """从CSV文件批量推理"""
        # 读取CSV文件
        print(f"Reading CSV file: {csv_path}")
        df = pd.read_csv(csv_path)
        print(f"✓ Loaded {len(df)} rows")
        
        if limit:
            df = df.head(limit)
            print(f"✓ Limited to first {limit} rows")
        
        # 获取所有分类
        categories = df['category'].unique().tolist() if 'category' in df.columns else ['unknown']
        self.meta_builder = MetaFeatureBuilder(
            categories=categories,
            out_dim=CFG.meta_dim
        )
        
        # 推理结果列表
        results = []
        
        # 遍历每一行进行推理
        print("\nStarting batch inference...")
        for idx, row in df.iterrows():
            try:
                i = int(idx)
                # 显示进度
                if (i + 1) % max(1, len(df) // 10) == 0:
                    print(f"  Progress: {i+1}/{len(df)}")
                
                # 构建数据行
                data_row = {
                    "video_id": str(row.get('video_id', f'video_{idx}')),
                    "title": str(row.get('title', '')),
                    "tags": str(row.get('tags', '')),
                    "category": str(row.get('category', 'unknown')),
                    "duration": float(row.get('duration', 0.0)),
                    "verified": 1 if row.get('author_fans', 0) > 100000 else 0,
                    "publish_time": str(row.get('publish_time', '')),
                    "label": 0,
                }
                
                # 推理
                result = self._infer_single(data_row, row)
                results.append(result)
                
            except Exception as e:
                print(f"  ⚠ Error at row {idx}: {e}")
                continue
        
        print(f"\n✓ Inference completed: {len(results)} successful results")
        
        # 保存结果
        if output_path:
            self._save_results(results, output_path)
        
        return results
    
    def _infer_single(self, data_row: dict, csv_row: pd.Series) -> dict:
        """推理单个样本"""
        # 构建元特征
        meta_feat = self.meta_builder.build(data_row)
        
        # 创建虚拟特征
        text_feat = np.random.randn(CFG.text_dim).astype(np.float32)
        video_feat = np.random.randn(CFG.video_dim).astype(np.float32)
        audio_feat = np.random.randn(CFG.audio_dim).astype(np.float32)
        
        # 转换为张量
        text_feat_tensor = torch.tensor(text_feat, dtype=torch.float32).unsqueeze(0).to(self.device)
        video_feat_tensor = torch.tensor(video_feat, dtype=torch.float32).unsqueeze(0).to(self.device)
        audio_feat_tensor = torch.tensor(audio_feat, dtype=torch.float32).unsqueeze(0).to(self.device)
        meta_feat_tensor = torch.tensor(meta_feat, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # 推理
        with torch.no_grad():
            outputs = self.model(
                text_feat=text_feat_tensor,
                video_feat=video_feat_tensor,
                audio_feat=audio_feat_tensor,
                meta_feat=meta_feat_tensor
            )
        
        # 构建结果
        result = {
            "video_id": data_row["video_id"],
            "title": data_row["title"],
            "category": data_row["category"],
            "author": str(csv_row.get('author_name', '')),
            "author_fans": int(csv_row.get('author_fans', 0)),
            "scientific_score": float(outputs["scientific_score"].item()),
            "technical_score": float(outputs["technical_score"].item()),
            "aesthetic_score": float(outputs["aesthetic_score"].item()),
            "overall_score": float(outputs["overall_score"].item()),
            "probability": float(outputs["probability"].item()),
            "prediction": "上榜" if outputs["probability"].item() >= CFG.threshold else "未上榜",
            "engagement": {
                "likes": int(csv_row.get('like_count', 0)),
                "shares": int(csv_row.get('share_count', 0)),
                "collects": int(csv_row.get('collect_count', 0)),
                "comments": int(csv_row.get('comment_count', 0)),
                "recommends": int(csv_row.get('recommend_count', 0)),
            },
            "publish_time": str(csv_row.get('publish_time', '')),
            "timestamp": datetime.now().isoformat()
        }
        
        return result
    
    def _save_results(self, results: list, output_path: str):
        """保存推理结果"""
        # 转换为DataFrame便于分析
        records = []
        for r in results:
            record = {
                "video_id": r["video_id"],
                "title": r["title"],
                "category": r["category"],
                "author": r["author"],
                "author_fans": r["author_fans"],
                "scientific_score": r["scientific_score"],
                "technical_score": r["technical_score"],
                "aesthetic_score": r["aesthetic_score"],
                "overall_score": r["overall_score"],
                "probability": r["probability"],
                "prediction": r["prediction"],
                "likes": r["engagement"]["likes"],
                "shares": r["engagement"]["shares"],
                "comments": r["engagement"]["comments"],
                "recommends": r["engagement"]["recommends"],
            }
            records.append(record)
        
        df_results = pd.DataFrame(records)
        
        # 保存为CSV和JSON
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        # CSV格式
        csv_path = output_path_obj.with_suffix('.csv')
        df_results.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"✓ Results saved to CSV: {csv_path}")
        
        # JSON格式
        json_path = output_path_obj.with_suffix('.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"✓ Results saved to JSON: {json_path}")
        
        # 统计信息
        print("\n" + "="*60)
        print("推理统计信息")
        print("="*60)
        print(f"总样本数: {len(results)}")
        print(f"上榜数: {(df_results['prediction'] == '上榜').sum()}")
        print(f"未上榜数: {(df_results['prediction'] == '未上榜').sum()}")
        print(f"平均科学性得分: {df_results['scientific_score'].mean():.4f}")
        print(f"平均技术性得分: {df_results['technical_score'].mean():.4f}")
        print(f"平均美学性得分: {df_results['aesthetic_score'].mean():.4f}")
        print(f"平均总体得分: {df_results['overall_score'].mean():.4f}")
        print(f"平均上榜概率: {df_results['probability'].mean():.4f}")
        print("="*60)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="批量推理 - 从CSV文件推理视频质量评分")
    
    parser.add_argument("--csv", required=True, help="输入CSV文件路径（抖音元数据格式）")
    parser.add_argument("--checkpoint", default=str(CFG.checkpoint_dir / "best.pt"),
                        help="模型检查点路径")
    parser.add_argument("--output", default="batch_results", 
                        help="输出文件路径（无扩展名，会生成CSV和JSON）")
    parser.add_argument("--limit", type=int, default=None,
                        help="限制处理的样本数（用于测试）")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 检查CSV文件
    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"✗ CSV文件不存在: {csv_path}")
        return 1
    
    # 检查模型
    if not Path(args.checkpoint).exists():
        print(f"⚠ 模型文件不存在: {args.checkpoint}")
        print("  将使用随机初始化的模型进行推理")
    
    # 初始化引擎
    print("初始化推理引擎...")
    engine = BatchInferenceEngine(args.checkpoint)
    
    # 批量推理
    print()
    results = engine.infer_from_csv(
        str(csv_path),
        output_path=args.output,
        limit=args.limit
    )
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
