"""全局配置"""
from pathlib import Path

# 项目根目录
BASE_DIR = Path(__file__).resolve().parent.parent

# 数据存储路径
DATA_DIR = BASE_DIR / "data"
TXT_DIR = DATA_DIR / "txt"
DB_PATH = DATA_DIR / "interview.db"

# 确保目录存在
TXT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# 反爬配置（Anti-Detection）
# ============================================================

# User-Agent 池 — 轮换使用，避免单一 UA 被标记
UA_POOL = [
    # Chrome (Windows)
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    # Chrome (Mac)
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    # Edge
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36 Edg/122.0.0.0",
    # Firefox
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:122.0) Gecko/20100101 Firefox/122.0",
]

# 完整浏览器请求头模板（更像真实浏览器）
BROWSER_HEADERS_TEMPLATE = {
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    "Accept-Encoding": "gzip, deflate, br",
    "Origin": "https://www.nowcoder.com",
    "Referer": "https://www.nowcoder.com/",
    "Sec-Ch-Ua": '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
    "Sec-Ch-Ua-Mobile": "?0",
    "Sec-Ch-Ua-Platform": '"Windows"',
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-site",
    "Connection": "keep-alive",
}

# ============================================================
# 爬虫配置
# ============================================================

CRAWLER = {
    # API 地址
    "search_api": "https://gw-c.nowcoder.com/api/sparta/pc/search",
    "discuss_api": "https://gw-c.nowcoder.com/api/sparta/detail/content-data/detail",
    "feed_url": "https://www.nowcoder.com/feed/main/detail",
    "discuss_url": "https://www.nowcoder.com/discuss",

    # 抓取范围
    "max_pages_per_query": 5,
    "max_posts_per_query": 30,
    "timeout": 15,
    # 正文有效长度阈值（字符）：短于该值视为无效面经，丢弃并继续翻页补充
    "min_content_length": 30,

    # ====== 反爬参数 ======
    # 基础请求间隔（秒）。真实延迟 = delay + 随机抖动(0~jitter)
    "delay": 1.2,
    "jitter": 0.8,           # 随机抖动上限（秒）
    # 失败重试
    "max_retries": 2,        # 单次请求最多重试次数
    "retry_base_delay": 3.0, # 重试基础等待（秒），实际 = base * 2^attempt（指数退避）
    # 风控检测关键词（响应体命中这些说明被风控了）
    "risk_keywords": [
        "滑块验证", "图形验证", "请稍后再试",
        "请求过于频繁", "操作太频繁", "访问受限",
        "403 Forbidden", "captcha", "风控",
    ],
}
