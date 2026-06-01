import pandas as pd
import re

# ========= 配置 =========
INPUT_FILE = "meta_data_03233.xlsx"   # ← 改成你的xlsx文件
OUTPUT_FILE = "parsed_metadata.csv"

# ========= 工具函数 =========
def parse_duration(duration_str):
    """'197.00秒' → 197.0"""
    if pd.isna(duration_str):
        return None
    match = re.search(r"([\d.]+)", str(duration_str))
    return float(match.group(1)) if match else None


def parse_int(value):
    """安全转 int"""
    try:
        return int(float(value))
    except:
        return None


def clean_text(text):
    if pd.isna(text):
        return ""
    return str(text).strip()


def extract_tags(desc):
    """从描述中提取 #标签"""
    if pd.isna(desc):
        return ""
    tags = re.findall(r"#\S+", desc)
    return " ".join(tags)


# ========= 主逻辑 =========
def process_excel(input_file):

    # === 读取 Excel ===
    df = pd.read_excel(input_file, engine="openpyxl")

    # === 标准列名（防止空格/隐藏字符）===
    df.columns = df.columns.str.strip()

    # === 列名映射（适配你截图）===
    column_map = {
        "博主名字": "author_name",
        "博主网址": "author_url",
        "博主粉丝量": "author_fans",
        "视频ID": "video_id",
        "视频描述": "video_desc",
        "话题标签": "tags",
        "发布时间": "publish_time",
        "视频链接": "video_url",
        "点赞量": "like_count",
        "转发量": "share_count",
        "收藏量": "collect_count",
        "评论量": "comment_count",
        "推荐量": "recommend_count",
        "时长": "duration",
        "类别": "category",
        "是否上榜": "is_top"
    }

    # === 重命名 ===
    df = df.rename(columns=column_map)

    # ========= 数据清洗 =========
    df["author_name"] = df["author_name"].apply(clean_text)
    df["author_url"] = df["author_url"].apply(clean_text)

    df["author_fans"] = df["author_fans"].apply(parse_int)
    df["video_id"] = df["video_id"].astype(str)

    df["video_desc"] = df["video_desc"].apply(clean_text)

    # 标签缺失自动补
    df["tags"] = df.apply(
        lambda row: row["tags"] if pd.notna(row["tags"]) and row["tags"] != ""
        else extract_tags(row["video_desc"]),
        axis=1
    )

    df["publish_time"] = pd.to_datetime(df["publish_time"], errors="coerce")

    df["video_url"] = df["video_url"].apply(clean_text)

    df["like_count"] = df["like_count"].apply(parse_int)
    df["share_count"] = df["share_count"].apply(parse_int)
    df["collect_count"] = df["collect_count"].apply(parse_int)
    df["comment_count"] = df["comment_count"].apply(parse_int)
    df["recommend_count"] = df["recommend_count"].apply(parse_int)

    df["duration"] = df["duration"].apply(parse_duration)

    df["category"] = df["category"].apply(clean_text)
    df["is_top"] = df["is_top"].apply(parse_int)

    # ========= 输出字段顺序 =========
    final_columns = [
        "author_name",
        "author_url",
        "author_fans",
        "video_id",
        "video_desc",
        "tags",
        "publish_time",
        "video_url",
        "like_count",
        "share_count",
        "collect_count",
        "comment_count",
        "recommend_count",
        "duration",
        "category",
        "is_top"
    ]

    df = df[final_columns]

    return df


# ========= 执行 =========
if __name__ == "__main__":
    df = process_excel(INPUT_FILE)

    print("解析结果：")
    print(df.head())

    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    print(f"\n已保存：{OUTPUT_FILE}")