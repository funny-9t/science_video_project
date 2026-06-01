# 🎬 科学视频质量评分系统 - 前端项目

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Status](https://img.shields.io/badge/status-active-brightgreen.svg)

一个基于**多模态深度学习**的科学短视频质量综合评估系统，提供现代化Web界面进行实时推理。

## ✨ 核心特性

### 🎯 多维度评分
- **🔬 科学性评分** - 内容的科学严谨性和准确性
- **⚙️ 技术性评分** - 视频/音频质量和制作水平
- **🎨 美学性评分** - 视觉设计和呈现效果
- **⭐ 总体得分** - 三维度综合加权评分

### 💻 用户友好的Web界面
- 响应式设计，支持桌面和移动设备
- 直观的表单输入和实时反馈
- 美观的可视化得分展示
- 结果导出为JSON格式

### 🚀 高效的推理引擎
- 支持GPU加速推理
- 模块化的编码器架构
- 灵活的特征处理流程
- 可扩展的API设计

### 📊 完整的元数据支持
- 基础信息：视频ID、标题、标签、分类
- 时间信息：发布时间、时间编码
- 互动数据：点赞、转发、评论
- 账户信息：认证状态

---

## 🏗️ 项目结构

```
science_video_project/
├── app.py                          # Flask Web应用主程序
├── requirements.txt                # Python依赖
├── frontend/                       # 前端文件
│   ├── index.html                 # Web界面
│   └── static/
│       ├── style.css              # 样式表
│       └── script.js              # 交互脚本
├── inference/
│   ├── infer.py                   # 基础推理脚本
│   └── full_infer.py              # 完整推理脚本
├── pipeline/
│   ├── config.py                  # 配置管理
│   ├── step_text.py               # 文本编码器
│   ├── step_video.py              # 视频编码器
│   ├── step_audio.py              # 音频编码器
│   ├── step_meta.py               # 元特征构建
│   ├── step_extract.py            # 视频提取
│   └── ...
├── training/
│   ├── model_mvp.py               # 模型定义
│   ├── train.py                   # 训练脚本
│   ├── dataset_pair.py            # 数据集
│   └── ...
└── outputs/
    ├── checkpoints/               # 模型检查点
    ├── features/                  # 特征缓存
    └── logs/                      # 训练日志
```

---

## 🚀 快速开始

### 前置要求
- Python 3.8+
- CUDA 11.8+ (可选，用于GPU加速)
- 足够的磁盘空间和内存

### 安装步骤

#### 1️⃣ 克隆或导航到项目目录
```bash
cd d:\Projects\science_video_ranker_mvp\science_video_project
```

#### 2️⃣ 创建虚拟环境（推荐）
```bash
python -m venv venv
.\venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac
```

#### 3️⃣ 安装依赖
```bash
pip install flask flask-cors
pip install -r requirements.txt
```

#### 4️⃣ 启动应用
```bash
python app.py
```

#### 5️⃣ 打开浏览器
访问 http://localhost:5000

---

## 📖 使用指南

### 基本工作流程

```
1. 填写视频基本信息
   ├─ 视频ID
   ├─ 标题
   ├─ 分类
   ├─ 标签
   └─ 其他元数据
         ↓
2. 输入互动数据（可选）
   ├─ 点赞数
   ├─ 转发数
   └─ 评论数
         ↓
3. 点击"开始评分推理"
   └─ 系统自动提取特征并推理
         ↓
4. 查看详细评分结果
   ├─ 三维度得分卡片
   ├─ 总体得分展示
   ├─ 上榜概率
   └─ 预测结果
         ↓
5. 导出结果为JSON
   └─ 保存评分数据
```

### 评分解读

| 分数范围 | 含义 | 等级 |
|---------|------|------|
| 0.8 - 1.0 | 优秀 | ⭐⭐⭐⭐⭐ |
| 0.6 - 0.8 | 良好 | ⭐⭐⭐⭐ |
| 0.4 - 0.6 | 中等 | ⭐⭐⭐ |
| 0.2 - 0.4 | 较差 | ⭐⭐ |
| 0.0 - 0.2 | 很差 | ⭐ |

### 预测说明

- **"上榜"** 
  - 概率 ≥ 50%
  - 表示视频质量优秀，值得推荐
  - 综合评分较高
  
- **"未上榜"**
  - 概率 < 50%
  - 表示需要进一步改进
  - 某些维度得分较低

---

## 🔌 API接口

### 推理端点

```http
POST /api/infer
Content-Type: application/json

{
    "video_id": "test_video_001",
    "title": "黑洞物理学入门",
    "tags": "黑洞|物理|天文",
    "category": "科普",
    "duration": 300,
    "verified": 1,
    "publish_time": "2024-04-29T14:30:00Z",
    "likes": 1500,
    "shares": 320,
    "comments": 580
}
```

**响应示例：**
```json
{
    "success": true,
    "video_id": "test_video_001",
    "scientific_score": 0.75,
    "technical_score": 0.82,
    "aesthetic_score": 0.68,
    "overall_score": 0.78,
    "probability": 0.85,
    "prediction": "上榜",
    "timestamp": "2024-04-29T14:30:00.123456"
}
```

### 健康检查端点

```http
GET /api/health
```

**响应：**
```json
{
    "status": "ready",
    "timestamp": "2024-04-29T14:30:00.123456"
}
```

---

## 🛠️ 配置说明

### 模型配置

编辑 `pipeline/config.py` 来修改默认配置：

```python
@dataclass
class Config:
    # 输出目录
    output_dir: Path = Path(r"F:\science_video_outputs")
    
    # 模型路径
    text_model_name: str = r"D:\...\chinese-robeta-wwm-ext"
    clip_model_name: str = r"D:\...\openaiclip-vit-large-patch14"
    asr_model_path: str = r"D:\...\faster-whisper-large-v2"
    
    # 特征维度
    text_dim: int = 768
    video_dim: int = 512
    audio_dim: int = 384
    meta_dim: int = 16
    
    # 设备和性能
    device: str = "cuda"  # 或 "cpu"
    threshold: float = 0.5
```

### 类别自定义

在 `app.py` 的 `init_models()` 函数中修改：

```python
_meta_builder = MetaFeatureBuilder(
    categories=["科技", "教育", "科普", "演讲", "实验", "评论", "其他"],
    out_dim=CFG.meta_dim
)
```

---

## 📊 特征系统

### 文本特征 (768维)
- 使用BERT类预训练模型
- 提取标题和标签的语义特征
- 基于RoBERTa-wwm中文模型

### 视频特征 (512维)
- 基于CLIP模型的视觉编码
- 从关键帧提取视觉特征
- ViT-Large-Patch14架构

### 音频特征 (384维)
- 使用Whisper特征提取
- 基于频谱分析和语义编码
- Faster-Whisper大模型

### 元特征 (16维)
- 8维数值特征：时长、标题长度、标签数等
- 8维时间编码：小时和周天的sin/cos变换
- 分类向量：One-hot编码

---

## 🔍 模型架构

```
输入特征
├─ text_feat (768)
├─ video_feat (512)
├─ audio_feat (384)
└─ meta_feat (16)
     ↓
三分支架构
├─ Scientific Branch
│  ├─ text_proj → MLP(768→128)
│  ├─ meta_proj → MLP(16→128)
│  └─ fusion → score_head → 科学性得分
├─ Technical Branch
│  ├─ video_proj → MLP(512→128)
│  ├─ audio_proj → MLP(384→128)
│  ├─ meta_proj → MLP(16→128)
│  └─ fusion → score_head → 技术性得分
└─ Aesthetic Branch
   ├─ video_proj → MLP(512→128)
   ├─ text_proj → MLP(768→128)
   ├─ audio_proj → MLP(384→128)
   └─ fusion → score_head → 美学性得分
     ↓
门控融合 (Gate Fusion)
├─ 学习三分支权重
├─ 加权融合
└─ 最终融合MLP
     ↓
输出层
├─ 科学性得分
├─ 技术性得分
├─ 美学性得分
├─ 总体得分
└─ 上榜概率 (Sigmoid)
```

---

## 💡 使用场景

### 场景1：内容审核
```
输入：待审视频信息
↓
系统评分：
  - 科学性 > 0.6? (检查是否科学准确)
  - 技术性 > 0.5? (检查是否制作精良)
↓
输出：通过/拒绝 + 改进建议
```

### 场景2：质量排序
```
输入：100个待审视频
↓
系统评分：为每个视频计算总体得分
↓
输出：按总体得分排序的视频列表
      + 分类推荐（上榜/未上榜）
```

### 场景3：改进指导
```
输入：用户的视频
↓
系统评分：获得三维度分数
↓
分析弱点：
  - 科学性低? → 改进内容准确性
  - 技术性低? → 提高视频/音频质量
  - 美学性低? → 改进视觉设计
↓
输出：针对性改进建议
```

---

## 📈 性能优化

### GPU加速
```python
# 在 config.py 中启用GPU
device: str = "cuda"

# 验证GPU是否可用
python -c "import torch; print(torch.cuda.is_available())"
```

### 批处理推理
```python
# 修改 app.py 支持批处理
# 收集多个请求后统一推理
```

### 特征缓存
```python
# 缓存已提取的特征以加速重复推理
# 避免重复的特征提取计算
```

---

## 🐛 故障排除

### 问题1：模型加载失败
```
Error: Checkpoint not found
```
✅ **解决方案**：确保 `best.pt` 存在于输出目录

### 问题2：内存不足
```
RuntimeError: CUDA out of memory
```
✅ **解决方案**：
- 改用CPU: `device: str = "cpu"`
- 减少特征维度
- 清理GPU内存

### 问题3：依赖缺失
```
ModuleNotFoundError: No module named 'transformers'
```
✅ **解决方案**：重新安装依赖
```bash
pip install -r requirements.txt
```

---

## 📚 文档

- [快速开始](./QUICK_START.md) - 5分钟上手指南
- [部署指南](./DEPLOYMENT_GUIDE.md) - 详细的部署和配置说明
- [API文档](./API.md) - 完整的API参考（如存在）

---

## 🤝 项目架构总览

```
┌─────────────────────────────────────────────────┐
│          Web浏览器 (前端界面)                    │
│    HTML + CSS + JavaScript 响应式设计            │
└──────────────────┬──────────────────────────────┘
                   │ HTTP/JSON
                   ↓
┌─────────────────────────────────────────────────┐
│          Flask Web服务 (app.py)                  │
│  - /api/infer - 推理接口                        │
│  - /api/health - 健康检查                       │
└──────────────────┬──────────────────────────────┘
                   │
       ┌───────────┼───────────┐
       ↓           ↓           ↓
    ┌─────┐   ┌─────┐   ┌─────┐
    │文本  │   │视频  │   │音频  │   元数据
    │编码  │   │编码  │   │编码  │   构建器
    │器    │   │器    │   │器    │
    └──┬──┘   └──┬──┘   └──┬──┘   └──┬──┘
       └────┬────┴────┬────┘    ┌────┘
            │         │         │
            ↓         ↓         ↓
     ┌─────────────────────────────┐
     │   多模态质量评分模型         │
     │  (MultiModalQualityModel)   │
     │  - 三分支架构               │
     │  - 门控融合                 │
     │  - 端到端优化               │
     └──────────────┬──────────────┘
                    │
          ┌─────────┴─────────┐
          ↓                   ↓
      ┌────────┐          ┌────────┐
      │三维得分│          │总体概率│
      └────────┘          └────────┘
```

---

## 📝 使用示例

### Python调用
```python
import requests

data = {
    "video_id": "demo_001",
    "title": "黑洞物理学",
    "tags": "黑洞|物理",
    "category": "科普",
    "duration": 300,
    "verified": 1,
    "publish_time": "2024-04-29T14:30:00Z"
}

response = requests.post("http://localhost:5000/api/infer", json=data)
result = response.json()

print(f"科学性: {result['scientific_score']:.2f}")
print(f"技术性: {result['technical_score']:.2f}")
print(f"美学性: {result['aesthetic_score']:.2f}")
print(f"预测: {result['prediction']}")
```

### curl命令
```bash
curl -X POST http://localhost:5000/api/infer \
  -H "Content-Type: application/json" \
  -d '{
    "video_id": "demo_001",
    "title": "黑洞物理学",
    "tags": "黑洞|物理",
    "category": "科普",
    "duration": 300,
    "verified": 1,
    "publish_time": "2024-04-29T14:30:00Z"
  }'
```

---

## 📋 TODO和改进计划

- [ ] 支持批量推理接口
- [ ] 实现用户账户和历史记录
- [ ] 添加数据可视化面板
- [ ] 支持模型微调
- [ ] 实现特征缓存系统
- [ ] 优化移动端界面
- [ ] 添加多语言支持

---

## 📄 许可证

MIT License - 详见 LICENSE 文件

---

## 👨‍💻 开发者

Science Video Ranker MVP - 多模态视频质量评分系统

---

## 🙏 致谢

- 感谢所有依赖库的开发者
- 特别感谢CLIP、Whisper、RoBERTa模型团队

---

## 📞 支持和反馈

如遇到任何问题，请：
1. 查看文档和FAQ
2. 检查错误日志
3. 验证配置和依赖
4. 联系技术支持

**祝您使用愉快！** 🎉
