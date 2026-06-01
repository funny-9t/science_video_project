"""
完整推理脚本 - 支持视频特征提取
可以处理实际的视频文件或使用预提取的特征
"""
import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

import torch
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.build_sample import build_sample
from pipeline.config import CFG
from pipeline.step_audio import AudioEncoder
from pipeline.step_extract import VideoExtractor
from pipeline.step_meta import MetaFeatureBuilder
from pipeline.step_text import TextEncoder
from pipeline.step_video import VideoEncoder
from pipeline.utils_io import ensure_dir, get_video_id, load_metadata, save_json
from training.model_mvp import MultiModalQualityModel
from training.utils_train import adapt_checkpoint_state_dict, get_device


class InferenceEngine:
    """推理引擎 - 封装模型和编码器"""
    
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
        
        # 初始化编码器
        asr_path = Path(CFG.asr_model_path).resolve()
        audio_path = Path(CFG.audio_model_path).resolve()
        
        self.text_encoder = TextEncoder(CFG.text_model_name, str(asr_path), self.device, CFG.local_files_only)
        self.video_encoder = VideoEncoder(CFG.clip_model_name, self.device)
        self.audio_encoder = AudioEncoder(str(audio_path), self.device, out_dim=CFG.audio_dim, shared_model=self.text_encoder.asr)
        
        self.video_extractor = VideoExtractor(audio_sr=CFG.audio_sr)
        
        print("✓ All components initialized")
    
    def infer_from_video(self, video_path: str, row_data: dict) -> dict:
        """从视频文件推理"""
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        video_id = get_video_id(video_path)
        
        # 创建临时目录
        ensure_dir(CFG.audio_dir)
        ensure_dir(CFG.frame_dir)
        
        # 提取音频和帧
        wav_path = CFG.audio_dir / f"{video_id}.wav"
        frame_path = CFG.frame_dir / video_id
        ensure_dir(frame_path)
        
        print(f"Extracting audio and frames...")
        self.video_extractor.extract_audio(video_path, wav_path)
        self.video_extractor.extract_frames(video_path, frame_path, fps=CFG.frame_fps)
        
        # 编码特征
        print(f"Encoding features...")
        text_result = self.text_encoder.encode_fields(
            wav_path=str(wav_path),
            title=str(row_data.get("title", "") or ""),
            tags=str(row_data.get("tags", "") or ""),
        )
        video_feat = self.video_encoder.encode_frames(frame_path)
        audio_feat = self.audio_encoder.encode(str(wav_path))
        
        return {
            "text_feat": text_result.embedding,
            "video_feat": video_feat,
            "audio_feat": audio_feat,
        }
    
    def infer_from_features(self, features: dict, row_data: dict) -> dict:
        """从预提取的特征推理"""
        # 验证特征维度
        if features["text_feat"].shape[-1] != CFG.text_dim:
            raise ValueError(f"Text feature dim mismatch: {features['text_feat'].shape[-1]} vs {CFG.text_dim}")
        if features["video_feat"].shape[-1] != CFG.video_dim:
            raise ValueError(f"Video feature dim mismatch: {features['video_feat'].shape[-1]} vs {CFG.video_dim}")
        if features["audio_feat"].shape[-1] != CFG.audio_dim:
            raise ValueError(f"Audio feature dim mismatch: {features['audio_feat'].shape[-1]} vs {CFG.audio_dim}")
        
        return features
    
    def predict(self, features: dict, row_data: dict) -> dict:
        """执行推理预测"""
        # 构建元特征
        meta_builder = MetaFeatureBuilder(
            categories=row_data.get("categories", ["未知"]),
            out_dim=CFG.meta_dim
        )
        meta_feat = meta_builder.build(row_data)
        
        # 转换为张量
        text_feat_tensor = torch.tensor(features["text_feat"], dtype=torch.float32).unsqueeze(0).to(self.device)
        video_feat_tensor = torch.tensor(features["video_feat"], dtype=torch.float32).unsqueeze(0).to(self.device)
        audio_feat_tensor = torch.tensor(features["audio_feat"], dtype=torch.float32).unsqueeze(0).to(self.device)
        meta_feat_tensor = torch.tensor(meta_feat, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # 推理
        with torch.no_grad():
            outputs = self.model(
                text_feat=text_feat_tensor,
                video_feat=video_feat_tensor,
                audio_feat=audio_feat_tensor,
                meta_feat=meta_feat_tensor
            )
        
        # 提取结果
        result = {
            "scientific_score": float(outputs["scientific_score"].item()),
            "technical_score": float(outputs["technical_score"].item()),
            "aesthetic_score": float(outputs["aesthetic_score"].item()),
            "overall_score": float(outputs["overall_score"].item()),
            "probability": float(outputs["probability"].item()),
            "prediction": "上榜" if outputs["probability"].item() >= CFG.threshold else "未上榜",
        }
        
        return result
    
    def full_inference(self, video_path: str, row_data: dict) -> dict:
        """完整推理流程"""
        # 提取特征
        features = self.infer_from_video(video_path, row_data)
        
        # 预测
        result = self.predict(features, row_data)
        result["video_id"] = row_data.get("video_id", "unknown")
        result["timestamp"] = datetime.now().isoformat()
        
        return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="科学视频质量评分推理")
    
    parser.add_argument("--mode", choices=["video", "features", "web"], default="video",
                        help="推理模式: video(从视频文件), features(从特征文件), web(Web API)")
    
    # Video模式参数
    parser.add_argument("--video", help="视频文件路径")
    parser.add_argument("--checkpoint", default=str(CFG.checkpoint_dir / "best.pt"),
                        help="检查点路径")
    
    # 元数据
    parser.add_argument("--video_id", help="视频ID")
    parser.add_argument("--title", default="", help="视频标题")
    parser.add_argument("--tags", default="", help="关键词（用|分隔）")
    parser.add_argument("--category", default="unknown", help="视频分类")
    parser.add_argument("--duration", type=float, default=0.0, help="视频时长")
    parser.add_argument("--verified", type=int, default=0, help="认证状态")
    parser.add_argument("--publish_time", default="", help="发布时间")
    
    # 输出
    parser.add_argument("--output", default="", help="输出JSON路径")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    if args.mode == "video":
        if not args.video or not args.checkpoint:
            print("Error: --video 和 --checkpoint 是必需的")
            return
        
        # 初始化引擎
        engine = InferenceEngine(args.checkpoint)
        
        # 构建行数据
        row_data = {
            "video_id": args.video_id or "unknown",
            "title": args.title,
            "tags": args.tags,
            "category": args.category,
            "duration": args.duration,
            "verified": args.verified,
            "publish_time": args.publish_time,
            "label": 0,
            "categories": [args.category],
        }
        
        # 推理
        print("开始推理...")
        result = engine.full_inference(args.video, row_data)
        
        # 输出结果
        print("\n" + "="*50)
        print("评分结果")
        print("="*50)
        print(f"视频ID: {result['video_id']}")
        print(f"科学性: {result['scientific_score']:.4f}")
        print(f"技术性: {result['technical_score']:.4f}")
        print(f"美学性: {result['aesthetic_score']:.4f}")
        print(f"总体: {result['overall_score']:.4f}")
        print(f"上榜概率: {result['probability']*100:.2f}%")
        print(f"预测: {result['prediction']}")
        print("="*50)
        
        # 保存结果
        if args.output:
            save_json(result, args.output)
            print(f"✓ 结果已保存到: {args.output}")
        else:
            print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
