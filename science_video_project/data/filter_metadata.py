"""筛选只有可用视频的 metadata"""
import pandas as pd
from pathlib import Path

meta = pd.read_csv('parsed_metadata.csv')
video_dir = Path('videos')
available = {p.stem.replace('douyin.wtf_douyin_', '') for p in video_dir.glob('*.mp4')}

filtered = meta[meta['video_id'].astype(str).isin(available)].copy()

print(f'原始: {len(meta)} 条')
print(f'可用视频: {len(available)} 个')
print(f'筛选后: {len(filtered)} 条')
print(f'  正样本: {(filtered["label"]==1).sum()}')
print(f'  负样本: {(filtered["label"]==0).sum()}')

filtered.to_csv('parsed_metadata_filtered.csv', index=False, encoding='utf-8-sig')
print('已保存: parsed_metadata_filtered.csv')
