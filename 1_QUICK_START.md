# 快速开始指南

## 🚀 一键启动

### 第一步：安装依赖

```bash
cd d:\Projects\science_video_ranker_mvp\science_video_project

# 创建虚拟环境（推荐）
python -m venv venv
.\venv\Scripts\activate

# 安装依赖
pip install flask flask-cors
pip install -r requirements.txt
```

### 第二步：启动服务

```bash
python app.py
```

您应该看到类似的输出：
```
✓ Loaded checkpoint from F:\science_video_outputs\checkpoints\best.pt
✓ All models initialized successfully
✓ 服务已启动
 * Running on http://127.0.0.1:5000
```

### 第三步：打开浏览器

访问 http://localhost:5000

---

## 📝 使用示例

### 示例1：评估教育类科普视频

**输入数据**
```
视频ID: edu_video_001
标题: 黑洞物理学入门讲座
分类: 教育
标签: 黑洞|物理|天文|讲座
时长: 1800秒（30分钟）
发布时间: 2024-04-29 14:30
认证账户: ✓ 是
点赞数: 2500
转发数: 450
评论数: 890
```

**预期结果**
- 科学性得分: ~0.80 (高科学含量)
- 技术性得分: ~0.75 (制作质量良好)
- 美学性得分: ~0.72 (讲座类通常美学分较低)
- 上榜概率: ~0.85 (优秀内容)

---

### 示例2：评估短科普视频

**输入数据**
```
视频ID: short_pop_001
标题: 3分钟了解量子纠缠
分类: 科普
标签: 量子|物理|科学|短视频
时长: 180秒（3分钟）
发布时间: 2024-04-29 10:15
认证账户: ✗ 否
点赞数: 5000
转发数: 1200
评论数: 2300
```

**预期结果**
- 科学性得分: ~0.68 (简化的内容)
- 技术性得分: ~0.80 (短视频通常精致)
- 美学性得分: ~0.85 (短视频重视视觉效果)
- 上榜概率: ~0.76 (较好的内容)

---

## 🎨 前端界面说明

### 表单字段详解

| 字段 | 类型 | 说明 | 示例 |
|------|------|------|------|
| 视频ID | 文本 | 唯一标识符 | `video_001` |
| 标题 | 文本 | 视频标题 | `黑洞最新发现` |
| 分类 | 下拉 | 7个预设分类 | `科技` |
| 关键词/标签 | 文本 | 用\|分隔 | `黑洞\|天文\|科学` |
| 时长(秒) | 数字 | 视频长度 | `300` |
| 发布时间 | 日期时间 | ISO格式 | `2024-04-29T14:30` |
| 认证账户 | 复选框 | 是否认证 | ☑ |
| 点赞数 | 数字 | 互动数据 | `1500` |
| 转发数 | 数字 | 互动数据 | `320` |
| 评论数 | 数字 | 互动数据 | `580` |

---

## 📊 结果解读

### 得分含义

#### 🔬 科学性得分 (0.0 - 1.0)
- **> 0.8**: 高度科学严谨，内容准确权威
- **0.6 - 0.8**: 科学内容充足，基本准确
- **0.4 - 0.6**: 科学性一般，可能有简化
- **< 0.4**: 科学含量较低或存在错误

#### ⚙️ 技术性得分 (0.0 - 1.0)
- **> 0.8**: 高清视频，专业音频，优秀剪辑
- **0.6 - 0.8**: 清晰度良好，音质不错
- **0.4 - 0.6**: 视音质一般，制作简洁
- **< 0.4**: 质量较差，需要改进

#### 🎨 美学性得分 (0.0 - 1.0)
- **> 0.8**: 精美设计，视觉冲击力强
- **0.6 - 0.8**: 设计美观，视觉效果好
- **0.4 - 0.6**: 设计简洁，视觉效果一般
- **< 0.4**: 视觉效果欠佳

#### ⭐ 总体得分
三个维度的加权综合评分，权重由模型自动学习确定。

#### 📈 上榜概率
- **> 70%**: 强烈推荐上榜
- **50% - 70%**: 推荐上榜
- **30% - 50%**: 犹豫不决
- **< 30%**: 不推荐上榜

---

## 🔗 API调用示例

### Python调用

```python
import requests
import json

# API端点
url = "http://localhost:5000/api/infer"

# 请求数据
payload = {
    "video_id": "test_001",
    "title": "黑洞物理学",
    "tags": "黑洞|物理|天文",
    "category": "科技",
    "duration": 300,
    "verified": 1,
    "publish_time": "2024-04-29T14:30:00Z"
}

# 发送请求
response = requests.post(url, json=payload)
result = response.json()

# 处理结果
if result['success']:
    print(f"科学性: {result['scientific_score']:.2f}")
    print(f"技术性: {result['technical_score']:.2f}")
    print(f"美学性: {result['aesthetic_score']:.2f}")
    print(f"总分: {result['overall_score']:.2f}")
    print(f"预测: {result['prediction']}")
```

### JavaScript调用

```javascript
const data = {
    video_id: "test_001",
    title: "黑洞物理学",
    tags: "黑洞|物理|天文",
    category: "科技",
    duration: 300,
    verified: 1,
    publish_time: "2024-04-29T14:30:00Z"
};

fetch('http://localhost:5000/api/infer', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json'
    },
    body: JSON.stringify(data)
})
.then(response => response.json())
.then(result => {
    console.log(`科学性: ${result.scientific_score.toFixed(2)}`);
    console.log(`技术性: ${result.technical_score.toFixed(2)}`);
    console.log(`美学性: ${result.aesthetic_score.toFixed(2)}`);
    console.log(`总分: ${result.overall_score.toFixed(2)}`);
})
.catch(error => console.error('Error:', error));
```

---

## 💾 导出结果

点击"📥 导出结果"按钮，系统将下载一个JSON文件，包含：

```json
{
  "video_id": "test_video_001",
  "timestamp": "2024-04-29 14:30:00",
  "prediction": "上榜",
  "scientific_score": 0.75,
  "technical_score": 0.82,
  "aesthetic_score": 0.68,
  "overall_score": 0.78,
  "probability": 0.85
}
```

---

## ⌨️ 快捷键

| 快捷键 | 功能 |
|--------|------|
| `Ctrl + Enter` | 提交表单 |
| `F5` | 刷新页面 |
| `Ctrl + S` | 保存结果 |

---

## 🔍 常见问题

### Q: 推理需要多长时间？
**A**: 通常需要3-10秒，取决于系统配置和GPU可用性。

### Q: 可以批量评分吗？
**A**: 当前版本为单个视频评分。批量功能正在开发中。

### Q: 特征数据如何生成？
**A**: 当前版本使用随机特征。完整版本需要上传实际视频文件。

### Q: 可以修改阈值吗？
**A**: 可以在 `config.py` 中修改 `threshold` 参数。

### Q: 支持哪些视频格式？
**A**: 完整版本支持 MP4、MKV、AVI、MOV 等常见格式。

---

## 📞 获取帮助

1. 查看 `DEPLOYMENT_GUIDE.md` 了解详细的部署说明
2. 检查终端输出的错误信息
3. 验证所有依赖是否正确安装

祝您使用愉快！🎉
