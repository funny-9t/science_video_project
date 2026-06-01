"""
Flask应用 - 科学视频质量评分推理服务
"""
import json
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
from pipeline.step_audio import AudioEncoder
from pipeline.step_extract import VideoExtractor
from pipeline.step_meta import MetaFeatureBuilder
from pipeline.step_text import TextEncoder
from pipeline.step_video import VideoEncoder
from pipeline.utils_io import ensure_dir, get_video_id
from training.model_mvp import MultiModalQualityModel
from training.utils_train import adapt_checkpoint_state_dict, get_device


app = Flask(__name__, 
            template_folder='frontend',
            static_folder='frontend/static')
CORS(app)

# 全局变量 - 缓存加载的模型和编码器
_model = None
_device = None
_text_encoder = None
_audio_encoder = None
_video_encoder = None
_meta_builder = None


def init_models():
    """初始化所有模型和编码器"""
    global _model, _device, _text_encoder, _audio_encoder, _video_encoder, _meta_builder
    
    try:
        # 获取设备
        _device = get_device(CFG.device)
        print(f"Using device: {_device}")
        
        # 加载多模态模型
        _model = MultiModalQualityModel(
            text_dim=CFG.text_dim,
            video_dim=CFG.video_dim,
            audio_dim=CFG.audio_dim,
            meta_dim=CFG.meta_dim,
            hidden_dim=CFG.hidden_dim,
        ).to(_device)
        
        checkpoint_path = CFG.checkpoint_dir / "best.pt"
        if checkpoint_path.exists():
            ckpt = torch.load(checkpoint_path, map_location=_device)
            if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                state_dict = ckpt["model_state_dict"]
            else:
                state_dict = ckpt
            _model.load_state_dict(adapt_checkpoint_state_dict(state_dict))
            print(f"✓ Loaded checkpoint from {checkpoint_path}")
        else:
            print(f"⚠ Checkpoint not found at {checkpoint_path}")
        
        _model.eval()
        
        # 初始化编码器
        asr_path = Path(CFG.asr_model_path).resolve()
        audio_path = Path(CFG.audio_model_path).resolve()
        
        _text_encoder = TextEncoder(CFG.text_model_name, str(asr_path), _device, CFG.local_files_only)
        _audio_encoder = AudioEncoder(str(audio_path), _device, out_dim=CFG.audio_dim, shared_model=_text_encoder.asr)
        _video_encoder = VideoEncoder(CFG.clip_model_name, _device)
        
        # 初始化元数据构建器（使用默认类别）
        _meta_builder = MetaFeatureBuilder(
            categories=["数字智能", "工程制造", "天文宇宙", "医学健康",
                        "自然地理", "动物植物", "农林畜牧", "生物基因", "综合"],
            out_dim=CFG.meta_dim
        )
        
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

        # --- 3) 提取特征 ---
        feature_source = "simulated"
        if has_file and _video_encoder is not None:
            try:
                # 保存上传的视频到临时目录
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
                feature_source = "real"
                print(f"  [upload] ✓ Features extracted from {video_path.name}")
            except Exception as ext_e:
                print(f"  [upload] ⚠ Feature extraction failed, falling back to simulated: {ext_e}")
                has_file = False

        if not has_file:
            # 使用模拟特征
            text_feat_np = np.random.randn(CFG.text_dim).astype(np.float32)
            video_feat_np = np.random.randn(CFG.video_dim).astype(np.float32)
            audio_feat_np = np.random.randn(CFG.audio_dim).astype(np.float32)

        # --- 4) 推理 ---
        t_t = torch.tensor(text_feat_np, dtype=torch.float32).unsqueeze(0).to(_device)
        v_t = torch.tensor(video_feat_np, dtype=torch.float32).unsqueeze(0).to(_device)
        a_t = torch.tensor(audio_feat_np, dtype=torch.float32).unsqueeze(0).to(_device)
        m_t = torch.tensor(meta_feat, dtype=torch.float32).unsqueeze(0).to(_device)

        with torch.no_grad():
            out = _model(text_feat=t_t, video_feat=v_t, audio_feat=a_t, meta_feat=m_t)

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
            "engagement": {
                "likes": int(data.get('like_count', data.get('likeCount', 0))),
                "shares": int(data.get('share_count', data.get('shareCount', 0))),
                "collects": int(data.get('collect_count', data.get('collectCount', 0))),
                "comments": int(data.get('comment_count', data.get('commentCount', 0))),
                "recommends": int(data.get('recommend_count', data.get('recommendCount', 0))),
            },
            "timestamp": datetime.now().isoformat()
        }
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
