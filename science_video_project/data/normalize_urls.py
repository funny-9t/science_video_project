"""
清理/标准化 CSV 中的抖音链接：
- https://www.iesdouyin.com/share/video/{id}/?... → https://www.douyin.com/video/{id}
- https://www.douyin.com/video/{id}?previous_page=xxx → https://www.douyin.com/video/{id}
- 保留原本就是标准格式的链接不变
"""

import csv
import re
import sys
from pathlib import Path

# ==================== 配置 ====================
CSV_INPUT = Path(__file__).parent / "已标注数据_已解析.csv"
CSV_OUTPUT = Path(__file__).parent / "已标注数据_已解析_clean.csv"
CACHE_FILE = Path(__file__).parent / ".short_link_cache.csv"

# 匹配抖音视频 ID
VIDEO_ID_RE = re.compile(r"/video/(\d+)")


def normalize_url(url: str) -> str:
    """将任意抖音链接标准化为 https://www.douyin.com/video/{id}"""
    url = url.strip()
    m = VIDEO_ID_RE.search(url)
    if m:
        return f"https://www.douyin.com/video/{m.group(1)}"
    return url  # 无法提取 ID，保持原样


def clean_cache_file():
    """清理缓存文件中的 URL"""
    if not CACHE_FILE.exists():
        print(f"缓存文件不存在: {CACHE_FILE}")
        return

    rows = []
    with open(CACHE_FILE, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2:
                row[1] = normalize_url(row[1])
            rows.append(row)

    with open(CACHE_FILE, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    print(f"已清理缓存文件: {len(rows)} 条")


def clean_csv():
    """清理 CSV 数据文件中的链接"""
    if not CSV_INPUT.exists():
        print(f"错误：输入文件不存在: {CSV_INPUT}")
        print("请先运行 resolve_short_links.py 生成已解析文件，"
              "或修改 CSV_INPUT 指向原始文件。")
        # 尝试回退到原始文件
        alt_input = Path(__file__).parent / "已标注数据.csv"
        if alt_input.exists():
            print(f"将使用原始文件: {alt_input}")
            return clean_csv_file(alt_input, CSV_OUTPUT)
        sys.exit(1)

    return clean_csv_file(CSV_INPUT, CSV_OUTPUT)


def clean_csv_file(input_path: Path, output_path: Path):
    """清理指定 CSV 文件中的链接"""
    with open(input_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)

    changed = 0
    for row in rows:
        for i, cell in enumerate(row):
            if "douyin.com" in cell:
                new_url = normalize_url(cell)
                if new_url != cell:
                    row[i] = new_url
                    changed += 1

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    print(f"处理完成！")
    print(f"  总行数:     {len(rows)}")
    print(f"  标准化链接: {changed}")
    print(f"  输出文件:   {output_path}")


def main():
    print("=" * 60)
    print("抖音链接标准化工具")
    print("=" * 60)

    # 1. 清理缓存
    clean_cache_file()

    # 2. 清理 CSV
    print()
    clean_csv()

    print("\n完成！")


if __name__ == "__main__":
    main()
