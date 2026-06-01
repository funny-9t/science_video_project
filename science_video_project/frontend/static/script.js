/**
 * 科学视频评分系统 — 前端交互脚本
 */

const API_BASE = window.location.origin;

// ── DOM 引用 ──────────────────────────────────────────
const form          = document.getElementById('inferenceForm');
const submitBtn     = document.getElementById('submitBtn');
const uploadZone    = document.getElementById('uploadZone');
const uploadInput   = document.getElementById('videoFile');
const uploadPlace   = document.getElementById('uploadPlaceholder');
const uploadFileInfo= document.getElementById('uploadFileInfo');
const uploadName    = document.getElementById('uploadFileName');
const uploadSize    = document.getElementById('uploadFileSize');
const statusDot     = document.getElementById('statusDot');
const statusText    = document.getElementById('statusText');
const alertError    = document.getElementById('alertError');
const alertMsg      = document.getElementById('alertErrorMsg');
const emptyState    = document.getElementById('emptyState');
const resultContent = document.getElementById('resultContent');
const exportBtn     = document.getElementById('exportBtn');
const resetBtn      = document.getElementById('resetResultBtn');

// 结果元素
const $ = id => document.getElementById(id);
const r = {
  verdict:   'rVerdict',
  id:        'rId',
  category:  'rCategory',
  source:    'rSource',
  time:      'rTime',
  sciNum:    'rSciNum',
  sciFill:   'rSciFill',
  tecNum:    'rTecNum',
  tecFill:   'rTecFill',
  aesNum:    'rAesNum',
  aesFill:   'rAesFill',
  overall:   'rOverallDisplay',
  ovrFill:   'rOvrFill',
  probVal:   'rProbValue',
  probFill:  'rProbFill',
  engSec:    'engSection',
  eLikes:    'eLikes',
  eShares:   'eShares',
  eCollects: 'eCollects',
  eComments: 'eComments',
  eRec:      'eRecommends',
};

// ── 初始化 ──────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  checkHealth();
  form.addEventListener('submit', handleSubmit);
  exportBtn.addEventListener('click', exportResults);
  resetBtn.addEventListener('click', () => {
    resultContent.style.display = 'none';
    emptyState.style.display = 'flex';
  });
  // 文件上传交互
  uploadInput.addEventListener('change', onFileSelect);
  uploadZone.addEventListener('dragover', () => uploadZone.classList.add('dragover'));
  uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('dragover'));
  uploadZone.addEventListener('drop', () => uploadZone.classList.remove('dragover'));
});

// ── 健康检查 ────────────────────────────────────────
async function checkHealth() {
  try {
    const res = await fetch(`${API_BASE}/api/health`);
    const d = await res.json();
    if (d.status === 'ready') {
      statusDot.className = 'status-dot ready';
      statusText.textContent = '服务就绪';
    } else {
      statusDot.className = 'status-dot loading';
      statusText.textContent = '初始化中…';
      setTimeout(checkHealth, 3000);
    }
  } catch {
    statusDot.className = 'status-dot error';
    statusText.textContent = '连接失败';
  }
}

// ── 文件选择交互 ──────────────────────────────────────
function onFileSelect() {
  const file = uploadInput.files[0];
  if (!file) {
    uploadPlace.style.display = 'flex';
    uploadFileInfo.classList.remove('visible');
    uploadZone.classList.remove('has-file');
    return;
  }
  uploadPlace.style.display = 'none';
  uploadFileInfo.classList.add('visible');
  uploadZone.classList.add('has-file');
  uploadName.textContent = file.name;
  uploadSize.textContent = `(${(file.size / 1024 / 1024).toFixed(1)} MB)`;
}

// ── 表单提交 ──────────────────────────────────────────
async function handleSubmit(e) {
  e.preventDefault();
  hideError();

  if (!form.checkValidity()) {
    showError('请填写所有必需字段');
    return;
  }

  // 收集为 FormData（支持文件上传）
  const fd = new FormData(form);
  // 日期转换
  const dt = new Date(fd.get('publishTime'));
  fd.set('publishTime', dt.toISOString());

  // 转为 loading
  submitBtn.classList.add('loading');
  submitBtn.disabled = true;

  try {
    const res = await fetch(`${API_BASE}/api/infer`, {
      method: 'POST',
      body: fd,   // 浏览器自动设 multipart/form-data
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.error || `HTTP ${res.status}`);
    }
    const result = await res.json();
    if (result.success) displayResults(result);
    else showError(result.error || '推理失败');
  } catch (err) {
    showError(`错误: ${err.message}`);
  } finally {
    submitBtn.classList.remove('loading');
    submitBtn.disabled = false;
  }
}

// ── 显示结果 ──────────────────────────────────────────
function displayResults(result) {
  // 切换面板
  emptyState.style.display = 'none';
  resultContent.style.display = 'block';

  // 元信息
  $(r.id).textContent = result.video_id;
  $(r.category).textContent = result.category || '—';
  $(r.time).textContent = new Date(result.timestamp).toLocaleString('zh-CN');

  // 特征来源标签
  const srcEl = $(r.source);
  if (result.feature_source === 'real') {
    srcEl.textContent = '📹 真实视频特征';
    srcEl.className = 'source-badge real';
  } else {
    srcEl.textContent = '⚡ 模拟特征';
    srcEl.className = 'source-badge sim';
  }

  // 上榜判断
  const verdictEl = $(r.verdict);
  const isUp = result.prediction === '上榜';
  verdictEl.textContent = result.prediction;
  verdictEl.className = 'verdict-value ' + (isUp ? 'recommend' : 'not-recommend');

  // 三维度得分
  setScore(r.sciNum, r.sciFill, result.scientific_score);
  setScore(r.tecNum, r.tecFill, result.technical_score);
  setScore(r.aesNum, r.aesFill, result.aesthetic_score);

  // 综合得分（大数字）
  const ov = result.overall_score;
  $(r.overall).textContent = ov.toFixed(2);
  animateBar(r.ovrFill, ov);

  // 上榜概率
  const prob = result.probability;
  $(r.probVal).textContent = (prob * 100).toFixed(1) + '%';
  animateBar(r.probFill, prob);

  // 互动数据
  const eng = result.engagement;
  if (eng && (eng.likes || eng.comments)) {
    $(r.engSec).classList.remove('hidden');
    $(r.eLikes).textContent    = fmtNum(eng.likes || 0);
    $(r.eShares).textContent   = fmtNum(eng.shares || 0);
    $(r.eCollects).textContent = fmtNum(eng.collects || 0);
    $(r.eComments).textContent = fmtNum(eng.comments || 0);
    $(r.eRec).textContent      = fmtNum(eng.recommends || 0);
  } else {
    $(r.engSec).classList.add('hidden');
  }

  // 滚动到结果
  setTimeout(() => {
    document.getElementById('resultPanel').scrollIntoView({ behavior: 'smooth', block: 'start' });
  }, 150);
}

function setScore(numId, fillId, score) {
  $(numId).textContent = score.toFixed(2);
  animateBar(fillId, score);
}

function animateBar(id, value) {
  const el = $(id);
  const pct = Math.max(0, Math.min(100, value * 100));
  el.style.width = '0%';
  void el.offsetWidth; // force reflow
  el.style.width = pct + '%';
}

// ── 导出 JSON ────────────────────────────────────────
function exportResults() {
  const data = {
    video_id:       $(r.id).textContent,
    category:       $(r.category).textContent,
    prediction:     $(r.verdict).textContent,
    scientific_score:  parseFloat($(r.sciNum).textContent),
    technical_score:   parseFloat($(r.tecNum).textContent),
    aesthetic_score:   parseFloat($(r.aesNum).textContent),
    overall_score:     parseFloat($(r.overall).textContent),
    probability:       parseFloat($(r.probVal).textContent) / 100,
    engagement: {
      likes:    parseInt($(r.eLikes).textContent) || 0,
      shares:   parseInt($(r.eShares).textContent) || 0,
      collects: parseInt($(r.eCollects).textContent) || 0,
      comments: parseInt($(r.eComments).textContent) || 0,
      recommends: parseInt($(r.eRec).textContent) || 0,
    },
    timestamp: new Date().toISOString(),
  };
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `sci_score_${data.video_id}_${new Date().toISOString().slice(0,10)}.json`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
  toast('结果已导出');
}

// ── 工具函数 ──────────────────────────────────────────
function hideError() {
  alertError.classList.remove('visible');
  alertMsg.textContent = '';
}
function showError(msg) {
  alertMsg.textContent = msg;
  alertError.classList.add('visible');
}

function fmtNum(n) {
  if (n >= 1_000_000) return (n / 1_000_000).toFixed(1) + 'M';
  if (n >= 1_000)     return (n / 1_000).toFixed(1) + 'K';
  return String(n);
}

let toastTimer;
function toast(msg) {
  const el = document.getElementById('toast');
  el.textContent = '✓ ' + msg;
  el.classList.add('show');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => el.classList.remove('show'), 2800);
}

// ── 键盘快捷键 ────────────────────────────────────────
document.addEventListener('keydown', e => {
  if (e.ctrlKey && e.key === 'Enter' && !submitBtn.disabled) {
    form.dispatchEvent(new Event('submit'));
  }
});
