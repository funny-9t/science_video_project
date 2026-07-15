"""
将 已标注数据.csv（两行表头格式）转换为训练所需的 metadata.csv 格式。

源格式列映射:
  链接 → 提取 video_id
  内容（视频描述） → title
  相关热榜热词 → tags
  类型 → category
  是否上榜 → label (是=1, 否=0)
  博主名字 → author_name
  粉丝 → author_fans
  点赞量 → like_count
  评论量 → comment_count
  转发量 → share_count
  收藏量 → collect_count
  推荐量 → recommend_count
  标注日期 → publish_time

细粒度评分 (1-5分) → 分支监督目标:
  科普信息量 + 选题重要性 + 科普通俗性 + 内容趣味性 → sci_target (ScientificBranch)
  低层视觉质量 + 听觉质量                         → tech_target (TechnicalBranch)
  视频美学质量                                    → aes_target (AestheticBranch)
"""

import re
import csv
import os
from pathlib import Path


def extract_video_id(url: str) -> str:
    """从抖音链接中提取 video_id"""
    if not url:
        return ""
    # 匹配 https://www.douyin.com/video/123456789
    m = re.search(r'/video/(\d+)', url)
    if m:
        return m.group(1)
    # 短链接无法直接提取，使用 URL 的 hash 作为备选
    m = re.search(r'v\.douyin\.com/([A-Za-z0-9_]+)', url)
    if m:
        return "short_" + m.group(1)
    return ""


def parse_fans(fans_str: str) -> str:
    """统一粉丝数格式，如 '104.7万' → 1047000"""
    if not fans_str:
        return "0"
    fans_str = str(fans_str).strip()
    if '万' in fans_str:
        try:
            return str(int(float(fans_str.replace('万', '')) * 10000))
        except ValueError:
            return "0"
    try:
        return str(int(float(fans_str)))
    except ValueError:
        return "0"


def main():
    src = Path(r"d:\Projects\science_video_ranker_mvp\已标注数据.csv")
    dst = Path(r"d:\Projects\science_video_ranker_mvp\science_video_project\data\parsed_metadata.csv")

    # 读取原始 CSV（跳过第2行子表头）
    with open(src, "r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if len(rows) < 3:
        print("数据行数不足")
        return

    header_row1 = rows[0]
    # header_row2 = rows[1]  # 子表头，跳过
    data_rows = rows[2:]

    # 打印原始列名，方便调试
    print("原始表头 (第1行):")
    for i, h in enumerate(header_row1):
        print(f"  [{i}] {h}")

    # 构建列索引映射
    col_map = {}
    for i, h in enumerate(header_row1):
        h_clean = h.strip()
        col_map[h_clean] = i

    # 目标输出列
    output_columns = [
        "author_name", "author_url", "author_fans",
        "video_id", "title", "tags", "publish_time", "video_url",
        "like_count", "share_count", "collect_count", "comment_count",
        "recommend_count", "duration", "category", "label",
        # 七项细粒度评分 (原始1-5分)
        "science_info", "topic_importance", "science_access", "content_interest",
        "visual_quality", "audio_quality", "video_aesthetics",
        # 三条分支的聚合监督目标 (归一化到[0,1])
        "sci_target", "tech_target", "aes_target",
    ]

    converted = []
    skipped = 0
    for row in data_rows:
        if not any(cell.strip() for cell in row):
            continue  # 跳过空行

        def get_val(key: str, default: str = "") -> str:
            idx = col_map.get(key)
            if idx is not None and idx < len(row):
                return row[idx].strip()
            return default

        url = get_val("链接")
        video_id = extract_video_id(url)

        if not video_id:
            skipped += 1
            continue

        label_raw = get_val("是否上榜")
        # 兼容新旧标签格式: "是"/"上榜视频" → 1, "否"/"该条未上榜" → 0
        if label_raw in ("是", "上榜视频"):
            label = "1"
        else:
            label = "0"

        # ── 七项细粒度评分 (列索引 8-14，固定位置，Row2 中的名称) ──
        # [8]科普信息量  [9]选题重要性  [10]科普通俗性  [11]内容趣味性
        # [12]低层视觉质量  [13]听觉质量  [14]视频美学质量
        def get_cell(idx: int, default: str = "0") -> str:
            if idx < len(row):
                return row[idx].strip()
            return default

        def get_score(idx: int) -> float:
            try:
                return float(get_cell(idx, "0"))
            except ValueError:
                return 0.0

        sci_info   = get_score(8)
        topic_imp  = get_score(9)
        sci_access = get_score(10)
        content_int = get_score(11)
        vis_qual   = get_score(12)
        aud_qual   = get_score(13)
        vid_aes    = get_score(14)

        # 分支聚合目标: 均值归一化到 [0, 1] (原始 1-5)
        sci_target  = round((sci_info + topic_imp + sci_access + content_int) / 4.0 / 5.0, 4)
        tech_target = round((vis_qual + aud_qual) / 2.0 / 5.0, 4)
        aes_target  = round(vid_aes / 5.0, 4)

        row_out = {
            "author_name": get_val("博主名字"),
            "author_url": "",
            "author_fans": parse_fans(get_val("粉丝")),
            "video_id": video_id,
            "title": get_val("内容（视频描述）").replace("\n", " "),
            "tags": get_val("相关热榜热词"),
            "publish_time": get_val("标注日期"),
            "video_url": url,
            "like_count": get_val("点赞量", "0"),
            "share_count": get_val("转发量", "0"),
            "collect_count": get_val("收藏量", "0"),
            "comment_count": get_val("评论量", "0"),
            "recommend_count": get_val("推荐量", "0"),
            "duration": "",
            "category": get_val("类型"),
            "label": label,
            # 七项细粒度原始分
            "science_info": str(sci_info),
            "topic_importance": str(topic_imp),
            "science_access": str(sci_access),
            "content_interest": str(content_int),
            "visual_quality": str(vis_qual),
            "audio_quality": str(aud_qual),
            "video_aesthetics": str(vid_aes),
            # 三条分支聚合目标
            "sci_target": str(sci_target),
            "tech_target": str(tech_target),
            "aes_target": str(aes_target),
        }
        converted.append(row_out)

    # 写入 CSV
    with open(dst, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=output_columns)
        writer.writeheader()
        writer.writerows(converted)

    pos_count = sum(1 for r in converted if r["label"] == "1")
    neg_count = sum(1 for r in converted if r["label"] == "0")
    has_scores = sum(1 for r in converted if float(r.get("science_info", "0")) > 0)

    print(f"\n转换完成!")
    print(f"  总行数: {len(data_rows)}")
    print(f"  成功: {len(converted)} 条")
    print(f"  跳过 (无法提取video_id): {skipped} 条")
    print(f"  正样本(上榜): {pos_count}")
    print(f"  负样本(未上榜): {neg_count}")
    print(f"  含细粒度评分: {has_scores} 条")
    print(f"  输出文件: {dst}")


if __name__ == "__main__":
    main()
