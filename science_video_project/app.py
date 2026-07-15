"""
Flask应用 - 科学视频质量评分推理服务 (Research Edition)

支持 MVP 模型 (MultiModalQualityModel) 和研究版模型 (ResearchModel)。
通过 --research 参数指定加载研究版 checkpoint。
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime

import torch
import numpy as np
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.config import CFG
from pipeline.step_aesthetic_clip import CLIPAestheticScorer, DEFAULT_PROMPTS
from pipeline.step_audio import AudioEncoder
from pipeline.step_extract import VideoExtractor
from pipeline.step_meta import MetaFeatureBuilder
from pipeline.step_science_features import ScienceFeatureExtractor
from pipeline.step_text import TextEncoder
from pipeline.step_video import VideoEncoder
from pipeline.utils_io import ensure_dir, get_video_id
from training.utils_train import adapt_checkpoint_state_dict, get_device

# 模型类型
_USE_RESEARCH = os.environ.get("RESEARCH_MODE", "0") == "1"
if _USE_RESEARCH:
    from training.model_research import ResearchModel as QualityModel
else:
    from training.model_mvp import MultiModalQualityModel as QualityModel


app = Flask(__name__,
            template_folder='frontend',
            static_folder='frontend/static')
CORS(app)

# 全局变量
_model = None
_device = None
_text_encoder = None
_audio_encoder = None
_video_encoder = None
_meta_builder = None
_sci_extractor = None  # §7 新增
_aes_scorer = None     # §8 新增 (用于 app 推理时提取 aes_feat)


def _extract_per_frame_features(video_encoder: VideoEncoder, frame_dir: str | Path) -> torch.Tensor:
    """提取逐帧 CLIP 特征 (不池化)。仅研究版推理使用。"""
    from PIL import Image

    frame_paths = sorted(Path(frame_dir).glob("*.jpg"))
    if not frame_paths:
        return torch.zeros(0, 512, dtype=torch.float32)

    feats = []
    with torch.no_grad():
        for p in frame_paths:
            img = Image.open(p).convert("RGB")
            if video_encoder.backend == "hf":
                inputs = video_encoder.processor(images=img, return_tensors="pt")
                pixel = inputs["pixel_values"].to(video_encoder.device, dtype=video_encoder.model.dtype)
                out = video_encoder.model(pixel_values=pixel)
                feat = out.image_embeds
            else:
                image = video_encoder.preprocess(img).unsqueeze(0).to(video_encoder.device)
                feat = video_encoder.model.encode_image(image)
            feat = feat / feat.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            feats.append(feat.squeeze(0).detach().cpu())

    return torch.stack(feats, dim=0).float()


def init_models():
    """初始化所有模型和编码器。支持 MVP 和 Research 两种模式。"""
    global _model, _device, _text_encoder, _audio_encoder, _video_encoder
    global _meta_builder, _sci_extractor, _aes_scorer

    try:
        _device = get_device(CFG.device)
        print(f"Using device: {_device}")

        # 加载模型
        if _USE_RESEARCH:
            _model = QualityModel(
                text_dim=CFG.text_dim, video_dim=CFG.video_dim,
                audio_dim=CFG.audio_dim, meta_dim=CFG.meta_dim,
                aes_dim=CFG.aes_dim, sci_hand_dim=CFG.sci_hand_dim,
                hidden_dim=CFG.hidden_dim,
                use_quality_head=CFG.use_quality_head,
                use_engagement_branch=CFG.use_engagement_branch,
                use_science_features=CFG.use_science_features,
                use_aesthetic_mlp=CFG.use_aesthetic_mlp,
                use_temporal_encoder=CFG.use_temporal_encoder,
                use_cross_modal_attention=CFG.use_cross_modal_attention,
                temporal_dim=CFG.temporal_dim,
                temporal_arch=CFG.temporal_arch,
                temporal_num_layers=CFG.temporal_num_layers,
                temporal_num_heads=CFG.temporal_num_heads,
                cross_modal_num_heads=CFG.cross_modal_num_heads,
                cross_modal_dropout=CFG.cross_modal_dropout,
            ).to(_device)
            ckpt_name = "research_best.pt"
        else:
            _model = QualityModel(
                text_dim=CFG.text_dim, video_dim=CFG.video_dim,
                audio_dim=CFG.audio_dim, meta_dim=CFG.meta_dim,
                hidden_dim=CFG.hidden_dim,
            ).to(_device)
            ckpt_name = "best.pt"

        checkpoint_path = CFG.checkpoint_dir / ckpt_name
        if checkpoint_path.exists():
            ckpt = torch.load(checkpoint_path, map_location=_device)
            if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                state_dict = ckpt["model_state_dict"]
            else:
                state_dict = ckpt
            state_dict = adapt_checkpoint_state_dict(state_dict)
            # 允许部分匹配加载
            model_dict = _model.state_dict()
            matched = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
            model_dict.update(matched)
            _model.load_state_dict(model_dict)
            print(f"✓ Loaded checkpoint from {checkpoint_path} ({len(matched)} keys)")
        else:
            print(f"⚠ Checkpoint not found at {checkpoint_path}")

        _model.eval()

        # 编码器
        asr_path = Path(CFG.asr_model_path).resolve()
        audio_path = Path(CFG.audio_model_path).resolve()

        _text_encoder = TextEncoder(CFG.text_model_name, str(asr_path), _device, CFG.local_files_only)
        _audio_encoder = AudioEncoder(str(audio_path), _device, out_dim=CFG.audio_dim, shared_model=_text_encoder.asr)
        _video_encoder = VideoEncoder(CFG.clip_model_name, _device)
        _meta_builder = MetaFeatureBuilder(
            categories=["数字智能", "工程制造", "天文宇宙", "医学健康",
                        "自然地理", "动物植物", "农林畜牧", "生物基因", "综合"],
            out_dim=CFG.meta_dim,
        )

        # §7 科学性特征提取器
        _sci_extractor = ScienceFeatureExtractor()
        # §8 美学打分器（仅研究版需要）
        if _USE_RESEARCH:
            _aes_scorer = CLIPAestheticScorer(prompts=DEFAULT_PROMPTS, device=CFG.device)

        print("✓ All models initialized successfully")
        return True

    except Exception as e:
        print(f"✗ Error initializing models: {e}")
        import traceback
        traceback.print_exc()
        return False


@app.route('/')
def index():
    """主页面"""
    return render_template('index.html')


@app.route('/api/infer', methods=['POST'])
def api_infer():
    """
    推理API端点 — 同时支持 JSON 数据 和 multipart 文件上传
    """
    try:
        if _model is None:
            return jsonify(success=False, error="模型未初始化"), 503

        # --- 1) 解析输入 ---
        has_file = False
        if request.content_type and 'multipart/form-data' in request.content_type:
            # 来自表单的文件上传
            form = request.form
            video_file = request.files.get('videoFile', None)
            has_file = video_file is not None and video_file.filename
            data = {
                'video_id': form.get('videoId', form.get('video_id', 'upload')),
                'title': form.get('title', ''),
                'tags': form.get('tags', ''),
                'category': form.get('category', 'unknown'),
                'duration': float(form.get('duration', 0.0)),
                'author_fans': int(form.get('authorFans', 0)),
                'like_count': int(form.get('likeCount', 0)),
                'share_count': int(form.get('shareCount', 0)),
                'collect_count': int(form.get('collectCount', 0)),
                'comment_count': int(form.get('commentCount', 0)),
                'recommend_count': int(form.get('recommendCount', 0)),
                'verified': int(form.get('verified', 0)),
                'publish_time': form.get('publishTime', ''),
            }
        else:
            # 纯 JSON 请求
            data = request.get_json() or {}
            has_file = False

        # 验证必需字段
        required_fields = ['video_id', 'title', 'tags', 'category']
        for field in required_fields:
            if field not in data or not data.get(field):
                return jsonify(success=False, error=f"缺少必需字段: {field}"), 400

        # --- 2) 构建行数据 ---
        row = {
            "video_id": str(data['video_id']),
            "title": str(data.get('title', '')),
            "tags": str(data.get('tags', '')),
            "category": str(data.get('category', 'unknown')),
            "duration": float(data.get('duration', 0.0)),
            "verified": 1 if data.get('author_fans', 0) > 100000 else data.get('verified', 0),
            "publish_time": str(data.get('publish_time', '')),
            "label": 0,
        }
        meta_feat = _meta_builder.build(row)

        # 拼接文本 (用于科学性特征)
        combined_text = " ".join([str(data.get('title', '')), str(data.get('tags', '')), str(data.get('subtitle', ''))])

        # --- 3) 提取特征 ---
        feature_source = "simulated"
        sci_hand_np = np.zeros(CFG.sci_hand_dim, dtype=np.float32)
        aes_feat_np = np.zeros(CFG.aes_dim, dtype=np.float32)
        frame_feats_np = np.zeros((0, CFG.video_dim), dtype=np.float32)

        if has_file and _video_encoder is not None:
            try:
                upload_dir = CFG.output_dir / "uploads"
                ensure_dir(upload_dir)
                video_path = upload_dir / video_file.filename
                video_file.save(str(video_path))

                video_id = get_video_id(video_path)
                wav_path = CFG.audio_dir / f"{video_id}.wav"
                frame_path = CFG.frame_dir / video_id
                ensure_dir(frame_path)

                print(f"  [upload] Extracting features from {video_path.name}...")
                _video_extractor = VideoExtractor(audio_sr=CFG.audio_sr)
                _video_extractor.extract_audio(video_path, wav_path)
                _video_extractor.extract_frames(video_path, frame_path, fps=CFG.frame_fps)

                text_result = _text_encoder.encode_fields(
                    wav_path=str(wav_path),
                    title=str(data.get('title', '') or ''),
                    tags=str(data.get('tags', '') or ''),
                )
                video_feat = _video_encoder.encode_frames(frame_path)
                audio_feat = _audio_encoder.encode(str(wav_path))

                text_feat_np = text_result.embedding
                video_feat_np = video_feat.numpy() if hasattr(video_feat, 'numpy') else video_feat
                audio_feat_np = audio_feat.numpy() if hasattr(audio_feat, 'numpy') else audio_feat

                # §7 手工科学性特征
                if _sci_extractor is not None:
                    sci_hand_np = _sci_extractor.extract_vector(
                        " ".join([str(data.get('title', '')), str(data.get('tags', '')), str(text_result.subtitle)])
                    ).numpy()

                # §8 美学特征 (CLIP prompt scoring)
                if _USE_RESEARCH and _aes_scorer is not None:
                    aes_result = _aes_scorer.score_frames_batched(frame_path)
                    dim_names = list(aes_result["dimensions"].keys())
                    aes_feat_np = np.array([aes_result["dimensions"][n] for n in dim_names], dtype=np.float32)

                # §9 时序帧特征 (仅研究版)
                if _USE_RESEARCH and CFG.use_temporal_encoder:
                    frame_feats = _extract_per_frame_features(_video_encoder, frame_path)
                    frame_feats_np = frame_feats.numpy()

                feature_source = "real"
                print(f"  [upload] ✓ Features extracted from {video_path.name}")
            except Exception as ext_e:
                print(f"  [upload] ⚠ Feature extraction failed, falling back to simulated: {ext_e}")
                has_file = False

        if not has_file:
            text_feat_np = np.random.randn(CFG.text_dim).astype(np.float32)
            video_feat_np = np.random.randn(CFG.video_dim).astype(np.float32)
            audio_feat_np = np.random.randn(CFG.audio_dim).astype(np.float32)

        # --- 4) 推理 ---
        t_t = torch.tensor(text_feat_np, dtype=torch.float32).unsqueeze(0).to(_device)
        v_t = torch.tensor(video_feat_np, dtype=torch.float32).unsqueeze(0).to(_device)
        a_t = torch.tensor(audio_feat_np, dtype=torch.float32).unsqueeze(0).to(_device)
        m_t = torch.tensor(meta_feat, dtype=torch.float32).unsqueeze(0).to(_device)

        # 研究版额外特征
        model_kwargs = {"text_feat": t_t, "video_feat": v_t, "audio_feat": a_t, "meta_feat": m_t}
        if _USE_RESEARCH:
            model_kwargs["aes_feat"] = torch.tensor(aes_feat_np, dtype=torch.float32).unsqueeze(0).to(_device)
            model_kwargs["sci_hand_feat"] = torch.tensor(sci_hand_np, dtype=torch.float32).unsqueeze(0).to(_device)
            if frame_feats_np.size > 0:
                model_kwargs["frame_features"] = torch.tensor(frame_feats_np, dtype=torch.float32).unsqueeze(0).to(_device)

        with torch.no_grad():
            out = _model(**model_kwargs)

        prob = float(out["probability"].item())
        result = {
            "success": True,
            "video_id": str(data['video_id']),
            "title": str(data.get('title', '')),
            "category": str(data.get('category', '')),
            "feature_source": feature_source,
            "scientific_score": float(out["scientific_score"].item()),
            "technical_score": float(out["technical_score"].item()),
            "aesthetic_score": float(out["aesthetic_score"].item()),
            "overall_score": float(out["overall_score"].item()),
            "probability": prob,
            "prediction": "上榜" if prob >= CFG.threshold else "未上榜",
        }

        # §12 研究版扩展输出
        if "quality_score" in out:
            result["quality_score"] = float(out["quality_score"].item())
        if "engagement_score" in out:
            result["engagement_score"] = float(out["engagement_score"].item())
        if "gate_weights" in out:
            gw = out["gate_weights"].squeeze(0).cpu().tolist()
            result["gate_weights"] = {
                "scientific": float(gw[0]),
                "technical": float(gw[1]),
                "aesthetic": float(gw[2]),
            }

        result["engagement"] = {
            "likes": int(data.get('like_count', data.get('likeCount', 0))),
            "shares": int(data.get('share_count', data.get('shareCount', 0))),
            "collects": int(data.get('collect_count', data.get('collectCount', 0))),
            "comments": int(data.get('comment_count', data.get('commentCount', 0))),
            "recommends": int(data.get('recommend_count', data.get('recommendCount', 0))),
        }
        result["timestamp"] = datetime.now().isoformat()

        return jsonify(result), 200

    except Exception as e:
        print(f"推理错误: {e}")
        import traceback
        traceback.print_exc()
        return jsonify(success=False, error=str(e)), 500
        import traceback
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    """健康检查"""
    status = "ready" if _model is not None else "initializing"
    return jsonify({
        "status": status,
        "timestamp": datetime.now().isoformat()
    }), 200


@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Not found"}), 404


@app.errorhandler(500)
def internal_error(e):
    return jsonify({"error": "Internal server error"}), 500


if __name__ == '__main__':
    print("初始化模型和编码器...")
    if init_models():
        print("✓ 服务已启动")
        app.run(debug=False, host='0.0.0.0', port=5000)
    else:
        print("✗ 无法启动服务 - 模型初始化失败")
        sys.exit(1)
