import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, pipeline
import ahocorasick
import re
import numpy as np

class ScienceVideoContinuousEvaluator:
    def __init__(self, feat_path, ner_path, sci_dict_path):
        self.device = 0 if torch.cuda.is_available() else -1
        
        # 加载特征提取与 NER 模型 [cite: 23, 69]
        self.feat_tokenizer = AutoTokenizer.from_pretrained(feat_path)
        self.feat_model = AutoModel.from_pretrained(feat_path).to("cuda" if self.device == 0 else "cpu")
        from transformers import AutoModelForTokenClassification, TokenClassificationPipeline
        ner_model = AutoModelForTokenClassification.from_pretrained(ner_path).to("cuda" if self.device == 0 else "cpu")
        ner_tokenizer = AutoTokenizer.from_pretrained(ner_path)
        self.ner_pipeline = TokenClassificationPipeline(
            model=ner_model,
            tokenizer=ner_tokenizer,
            aggregation_strategy="simple",
            device=self.device
        )
        
        # 初始化 AC 自动机词典 [cite: 9]
        self.ac = ahocorasick.Automaton()
        with open(sci_dict_path, 'r', encoding='utf-8') as f:
            for word in f:
                word = word.strip()
                if len(word) > 1:
                    self.ac.add_word(word, word)
        self.ac.make_automaton()

    def _get_vector(self, text):
        """输出 768 维全局语义 embedding [cite: 23, 69]"""
        inputs = self.feat_tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
        inputs = {k: v.to(self.feat_model.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.feat_model(**inputs)
        return outputs.last_hidden_state[0, 0, :].cpu().numpy()

    def evaluate_continuous(self, description, tags, subtitles):
        """
        输出 0-100% 的连续评分结果 [cite: 12, 33]
        """
        # 1. 计算信息密度得分 (0-100%)
        # 结合 AC 词典匹配与模型 NER 识别 [cite: 9, 23]
        ac_terms = [val for _, val in self.ac.iter(subtitles)]
        ner_res = self.ner_pipeline(subtitles)
        ner_terms = [e['word'].replace(" ", "") for e in ner_res if e['score'] > 0.4]
        
        all_terms = list(set(ac_terms + ner_terms))
        # 密度计算：实体总长度占文本总长度比例，限制最大值为 100%
        raw_density = sum(len(t) for t in all_terms) / len(subtitles) if subtitles else 0
        info_score_percent = min(100.0, raw_density * 100) 

        # 2. 计算语义连贯性得分 (0-100%)
        # 基于句子间语义相似度 [cite: 30, 53]
        sentences = [s for s in re.split(r'[\t\n]', subtitles) if len(s) > 5]
        logic_score_percent = 50.0 # 默认值
        
        if len(sentences) >= 2:
            s_vecs = [self._get_vector(s[:100]) for s in sentences[:380]]
            sims = [F.cosine_similarity(torch.from_numpy(s_vecs[i]).unsqueeze(0), 
                                       torch.from_numpy(s_vecs[i+1]).unsqueeze(0)).item() 
                    for i in range(len(s_vecs)-1)]
            # 将相似度 (-1 到 1) 映射到 0-100
            avg_sim = np.mean(sims)
            logic_score_percent = max(0.0, min(100.0, (avg_sim + 1) / 2 * 100))

        # 3. 生成用于多模态融合的 Embedding [cite: 84, 86]
        global_vec = self._get_vector(f"{description} {tags} {subtitles[:200]}")

        return {
            "text_embedding": global_vec,  # 768维特征向量 [cite: 23]
            "detected_terms": all_terms,   # 识别出的术语列表 [cite: 39]
            "final_scores": {
                "information_density_percent": f"{info_score_percent:.2f}%", # 连续值输出 [cite: 12]
                "logic_coherence_percent": f"{logic_score_percent:.2f}%"     # 连续值输出 [cite: 12]
            }
        }

# ================= 运行测试 =================
import os
import glob

if __name__ == "__main__":
    evaluator = ScienceVideoContinuousEvaluator(
        feat_path=r"D:\Projects\SciBert\chinese-robeta-wwm-ext",
        ner_path=r"D:\Projects\SciBert\roberta-base-finetuned-cluener2020-chinese",
        sci_dict_path="sci_whitelist.txt"
    )
    
    # 定义存放 txt 字幕的文件夹
    input_dir = r"D:\Projects\SciBert\wav_text\videos\asr_output\txt" 
    
    # 获取文件夹下所有 .txt 文件
    txt_files = glob.glob(os.path.join(input_dir, "*.txt"))
    
    for file_path in txt_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        res = evaluator.evaluate_continuous(
            description="批量处理", 
            tags="科普", 
            subtitles=content[:500]
        )
                
        print(f"文件名: {os.path.basename(file_path)}")

        print(f"信息密度得分: {res['final_scores']['information_density_percent']}")
        print(f"逻辑连贯得分: {res['final_scores']['logic_coherence_percent']}")
        print(f"识别到的专业术语: {res['detected_terms']}")
        print("-" )
        