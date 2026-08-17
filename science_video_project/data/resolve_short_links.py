"""
解析 CSV 文件中抖音短链接 (v.douyin.com) → 长链接 (www.douyin.com/video/{id})
通过 HTTP 重定向获取真实的视频 ID 长链接。
"""

import csv
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional

import requests

# ==================== 配置 ====================
CSV_INPUT = Path(__file__).parent / "已标注数据.csv"
CSV_OUTPUT = Path(__file__).parent / "已标注数据_已解析.csv"
CACHE_FILE = Path(__file__).parent / ".short_link_cache.csv"

# 请求间隔（秒），避免被反爬
REQUEST_DELAY = 0.8
# 请求超时（秒）
REQUEST_TIMEOUT = 15
# User-Agent，模拟浏览器
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/150.0.0.0 Safari/537.36 Edg/150.0.0.0"
    ),
}

# 短链接正则：匹配 https://v.douyin.com/xxx/ （可能尾部有空格）
SHORT_LINK_RE = re.compile(r"https?://v\.douyin\.com/[A-Za-z0-9_\-]+/?\s*")

# 从各种抖音 URL 格式中提取视频 ID
VIDEO_ID_RE = re.compile(r"/video/(\d+)")


def extract_video_id(url: str) -> Optional[str]:
    """从抖音 URL 中提取视频 ID，支持 douyin.com 和 iesdouyin.com"""
    m = VIDEO_ID_RE.search(url)
    return m.group(1) if m else None


def normalize_douyin_url(url: str) -> Optional[str]:
    """
    将各种抖音链接格式统一为标准长链接：
    - https://www.douyin.com/video/{id}?xxx  → https://www.douyin.com/video/{id}
    - https://www.iesdouyin.com/share/video/{id}/?... → https://www.douyin.com/video/{id}
    """
    video_id = extract_video_id(url)
    if video_id:
        return f"https://www.douyin.com/video/{video_id}"
    return None


def load_cache() -> dict[str, str]:
    """加载已有的短链→长链映射缓存"""
    cache: dict[str, str] = {}
    if CACHE_FILE.exists():
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 2:
                    cache[row[0].strip()] = row[1].strip()
    return cache


def save_cache(cache: dict[str, str]) -> None:
    """保存短链→长链映射缓存"""
    with open(CACHE_FILE, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        for short, long in cache.items():
            writer.writerow([short, long])


def resolve_short_link(short_url: str, cache: dict[str, str], session: requests.Session) -> Optional[str]:
    """
    通过 HTTP 请求获取短链接的重定向目标。
    返回长链接字符串，失败返回 None。
    """
    # 清理尾部空格和斜杠
    short_url = short_url.strip()

    # 查缓存
    if short_url in cache:
        print(f"  [缓存命中] {short_url} → {cache[short_url]}")
        return cache[short_url]

    try:
        # HEAD 请求先试探，避免下载大体积页面
        resp = session.head(
            short_url,
            headers=HEADERS,
            timeout=REQUEST_TIMEOUT,
            allow_redirects=True,
        )
        # 有些 CDN 对 HEAD 返回 405，回退用 GET（限制只取头几 KB）
        if resp.status_code == 405:
            resp = session.get(
                short_url,
                headers=HEADERS,
                timeout=REQUEST_TIMEOUT,
                allow_redirects=True,
                stream=True,
            )
            # 只读一点数据触发重定向完成即可
            for _ in resp.iter_content(chunk_size=1):
                break
            resp.close()

        final_url = resp.url
        # 标准化为 www.douyin.com/video/{id} 格式
        clean_url = normalize_douyin_url(final_url)
        if clean_url:
            print(f"  [解析成功] {short_url} → {clean_url}")
            cache[short_url] = clean_url
            return clean_url
        else:
            print(f"  [警告] 无法从重定向地址提取视频ID: {final_url[:100]}")
            return None

    except requests.RequestException as e:
        print(f"  [失败] {short_url} → {e}")
        return None


def extract_short_links(text: str) -> list[str]:
    """从文本中提取所有 v.douyin.com 短链接"""
    return SHORT_LINK_RE.findall(text)


def main():
    if not CSV_INPUT.exists():
        print(f"错误：找不到输入文件 {CSV_INPUT}")
        sys.exit(1)

    # 加载缓存
    cache = load_cache()
    print(f"已加载 {len(cache)} 条缓存记录")

    # 读取原始 CSV（保留所有格式）
    with open(CSV_INPUT, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)

    total_rows = len(rows)
    short_link_count = 0
    resolved_count = 0
    failed_count = 0

    session = requests.Session()

    # 遍历每一行，查找并替换短链接
    for row_idx, row in enumerate(rows):
        for col_idx, cell in enumerate(row):
            short_links = extract_short_links(cell)
            if not short_links:
                continue

            for short_link in short_links:
                short_link_count += 1
                clean_short = short_link.strip()
                print(f"\n[{resolved_count + failed_count + 1}] 行{row_idx + 1}/{total_rows} 短链接: {clean_short[:50]}...")

                long_url = resolve_short_link(clean_short, cache, session)
                if long_url:
                    # 在单元格中替换短链接为长链接
                    row[col_idx] = row[col_idx].replace(short_link, long_url)
                    resolved_count += 1
                else:
                    failed_count += 1

                # 每次解析后立即保存缓存，支持断点续传
                save_cache(cache)

                # 请求间隔
                time.sleep(REQUEST_DELAY)

    # 写入输出文件
    with open(CSV_OUTPUT, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    # 保存缓存
    save_cache(cache)

    print(f"\n{'=' * 60}")
    print(f"处理完成！")
    print(f"  总行数:       {total_rows}")
    print(f"  发现短链接:   {short_link_count}")
    print(f"  解析成功:     {resolved_count}")
    print(f"  解析失败:     {failed_count}")
    print(f"  缓存条目:     {len(cache)}")
    print(f"  输出文件:     {CSV_OUTPUT}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
