# 科学视频质量评分系统 - 前端界面部署指南

## 📋 项目概览

本系统提供了一个现代化的Web前端界面，用于对科学短视频进行多维度质量评分。

### 功能特性
- 🎨 **响应式设计**：支持桌面和移动设备
- 🚀 **实时推理**：基于多模态深度学习的智能评分
- 📊 **可视化结果**：美观的评分卡片和图表展示
- 💾 **结果导出**：支持JSON格式导出评分结果
- 📱 **用户友好**：直观的表单设计和交互反馈

### 评分维度
1. **🔬 科学性** - 内容的科学严谨性和准确性
2. **⚙️ 技术性** - 视频/音频质量和制作水平
3. **🎨 美学性** - 视觉设计和呈现效果
4. **⭐ 总体得分** - 三个维度的综合评分

---

## 🛠️ 安装依赖

### 前置要求
- Python 3.8+
- CUDA 11.8+ (可选，用于GPU加速)

### 安装步骤

```bash
cd d:\Projects\science_video_ranker_mvp\science_video_project

# 1. 创建虚拟环境（推荐）
python -m venv venv
.\venv\Scripts\activate

# 2. 升级pip
python -m pip install --upgrade pip

# 3. 安装依赖
pip install flask flask-cors
pip install -r requirements.txt

# 如果使用GPU（可选）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

## 🚀 启动应用

### 方式一：直接运行

```bash
# 激活虚拟环境
cd d:\Projects\science_video_ranker_mvp\science_video_project
.\venv\Scripts\activate

# 运行应用
python app.py
```

应用将在 `http://localhost:5000` 启动

### 方式二：使用Gunicorn（生产环境）

```bash
pip install gunicorn

gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

---

## 💻 Web界面使用

### 输入表单

#### 必填字段
- **视频ID** - 视频的唯一标识符
- **标题** - 视频标题
- **分类** - 选择视频分类（科技、教育、科普、演讲、实验、评论、其他）
- **关键词/标签** - 用 `|` 分隔的多个标签，如 `天文|黑洞|科学`

#### 可选字段
- **时长** - 视频时长（秒）
- **发布时间** - 发布日期和时间
- **认证账户** - 是否为认证账户
- **互动数据** - 点赞数、转发数、评论数

### 推理流程

1. 填写表单信息
2. 点击 "🚀 开始评分推理" 按钮
3. 等待模型处理（通常需要几秒钟）
4. 查看详细的评分结果
5. 导出JSON格式的结果

### 结果解读

#### 得分范围
- 每个维度的得分范围为 **0.0 - 1.0**
- 更高的分数表示该维度表现更好

#### 预测结果
- **"上榜"** - 综合概率≥50%，表示视频质量优秀
- **"未上榜"** - 综合概率<50%，表示需要改进

#### 上榜概率
- 显示模型对"上榜"判定的置信度
- 基于三个维度的综合评估

---

## 📡 API文档

### 基础URL
```
http://localhost:5000
```

### 端点：推理API

**请求**
```
POST /api/infer
Content-Type: application/json

{
    "video_id": "test_video_001",
    "title": "宇宙黑洞最新发现揭示",
    "tags": "天文|黑洞|宇宙|科学",
    "category": "科普",
    "duration": 180,
    "verified": 1,
    "publish_time": "2024-04-29T14:30:00Z",
    "likes": 1500,
    "shares": 320,
    "comments": 580
}
```

**响应**
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

### 端点：健康检查

**请求**
```
GET /api/health
```

**响应**
```json
{
    "status": "ready",
    "timestamp": "2024-04-29T14:30:00.123456"
}
```

---

## 🔧 配置说明

### 修改模型路径

编辑 `science_video_project/pipeline/config.py`：

```python
@dataclass
class Config:
    # 输出目录
    output_dir: Path = Path(r"F:\science_video_outputs")
    
    # 模型路径
    text_model_name: str = r"D:\Projects\science_video_ranker_mvp\chinese-robeta-wwm-ext"
    clip_model_name: str = r"D:\Projects\science_video_ranker_mvp\openaiclip-vit-large-patch14"
    asr_model_path: str = str(project_root.parent / "faster-whisper-large-v2")
    
    # 设备选择
    device: str = "cuda"  # 或 "cpu"
```

### 自定义类别

在 `app.py` 中修改 `init_models()` 函数：

```python
_meta_builder = MetaFeatureBuilder(
    categories=["科技", "教育", "科普", "演讲", "实验", "评论", "其他"], 
    out_dim=CFG.meta_dim
)
```

---

## 🐛 故障排除

### 问题1：模块导入错误
```
ModuleNotFoundError: No module named 'transformers'
```

**解决方案**：
```bash
pip install transformers torch safetensors
```

### 问题2：模型文件未找到
```
FileNotFoundError: ASR model path not found
```

**解决方案**：检查配置文件中的模型路径是否正确

### 问题3：CUDA内存不足
```
RuntimeError: CUDA out of memory
```

**解决方案**：
- 在 `config.py` 中改用CPU: `device: str = "cpu"`
- 或减小批处理大小

### 问题4：端口被占用
```
OSError: [Errno 48] Address already in use
```

**解决方案**：
```bash
# 修改端口号
python app.py --port 5001

# 或杀死占用进程
lsof -ti:5000 | xargs kill -9
```

---

## 📊 特征说明

### 文本特征（768维）
- 使用BERT类模型提取标题和标签的语义特征
- 基于预训练的中文RoBERTa-wwm模型

### 视频特征（512维）
- 使用CLIP模型从视频帧提取视觉特征
- 基于OpenAI CLIP ViT-Large-Patch14

### 音频特征（384维）
- 使用Whisper模型的特征提取
- 基于Faster-Whisper大模型

### 元特征（16维）
- 视频时长（归一化）
- 标题长度（归一化）
- 标签数量（归一化）
- 认证状态
- 发布时间的时间编码（sin/cos变换）
- 分类向量（One-hot编码）

---

## 🎯 性能优化建议

1. **使用GPU加速**：确保CUDA正确安装
2. **模型量化**：使用INT8或FP16以减少内存占用
3. **批处理**：在大规模推理时使用批处理
4. **缓存特征**：对重复视频缓存提取的特征

---

## 📝 日志和调试

### 启用调试模式

编辑 `app.py` 的最后一行：
```python
app.run(debug=True, host='0.0.0.0', port=5000)
```

### 查看请求日志

```bash
# 在终端中查看Flask的请求日志
# Flask会自动打印所有HTTP请求和错误
```

---

## 🚀 部署到云服务

### Docker部署

1. 创建 `Dockerfile`：
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt flask flask-cors
COPY . .
CMD ["python", "app.py"]
```

2. 构建镜像：
```bash
docker build -t science-video-ranker .
```

3. 运行容器：
```bash
docker run -p 5000:5000 science-video-ranker
```

---

## 📞 技术支持

如遇到问题，请检查：
1. Python版本是否≥3.8
2. 所有依赖是否正确安装
3. 模型文件路径是否正确
4. 系统是否有足够的磁盘空间和内存

---

## 📄 许可证

本项目遵循开源许可证。详见项目根目录的LICENSE文件。
