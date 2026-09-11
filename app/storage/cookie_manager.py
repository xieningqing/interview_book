"""
Cookie 管理模块
负责加载、保存、注入牛客网 Cookie
"""
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from app.config import DATA_DIR

logger = logging.getLogger(__name__)

COOKIE_FILE = DATA_DIR / "cookies.json"

# 这类是追踪/统计 cookie，反爬不依赖它们，排除掉
DENY_PREFIXES = (
    "_saas_",     # 神策埋点
    "SENSOR_DATA",
    "sojourner",
    "__tea__",    # 字节/神策相关
)


def load_cookies() -> dict:
    """从 cookies.json 加载 Cookie，返回 {name: value} dict"""
    if not COOKIE_FILE.exists():
        return {}

    try:
        data = json.loads(COOKIE_FILE.read_text(encoding="utf-8"))

        # 支持三种格式:
        # 1. {"name": "xxx", "value": "yyy"}  —— 直接 dict
        # 2. [{"name": "xxx", "value": "yyy"}, ...]  —— 浏览器导出的 list
        # 3. "cookie1=value1; cookie2=value2"  —— header 字符串

        if isinstance(data, str):
            return parse_cookie_string(data)

        if isinstance(data, list):
            result = {}
            for item in data:
                if isinstance(item, dict) and "name" in item and "value" in item:
                    name, value = item["name"], item.get("value", "")
                    if not _is_denied(name):
                        result[name] = value
            return result

        if isinstance(data, dict):
            result = {}
            for name, value in data.items():
                if isinstance(name, str) and not _is_denied(name):
                    result[name] = value
            return result

        return {}
    except Exception as e:
        logger.warning(f"加载 cookies.json 失败: {e}")
        return {}


def parse_cookie_string(cookie_str: str) -> dict:
    """解析浏览器复制的 Cookie header 字符串: name1=val1; name2=val2"""
    result = {}
    for part in cookie_str.split(";"):
        part = part.strip()
        if not part or "=" not in part:
            continue
        name, _, value = part.partition("=")
        name = name.strip()
        value = value.strip()
        if name and not _is_denied(name):
            result[name] = value
    return result


def save_cookies(cookies: dict | str | list) -> dict:
    """
    保存 Cookie 到文件，返回标准化后的 dict

    输入支持:
    - dict: {"name": "value", ...}
    - str:  "cookie1=val1; cookie2=val2"
    - list: [{"name": "...", "value": "..."}, ...]
    """
    normalized = {}

    if isinstance(cookies, str):
        normalized = parse_cookie_string(cookies)
    elif isinstance(cookies, dict):
        for name, value in cookies.items():
            if isinstance(name, str):
                normalized[name] = str(value)
    elif isinstance(cookies, list):
        for item in cookies:
            if isinstance(item, dict) and "name" in item:
                normalized[item["name"]] = item.get("value", "")

    COOKIE_FILE.parent.mkdir(parents=True, exist_ok=True)
    COOKIE_FILE.write_text(json.dumps(normalized, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"✓ Cookie 已保存（{len(normalized)} 个）")
    return normalized


def mask_cookies(cookies: dict, show_chars: int = 6) -> dict:
    """对 Cookie 值做脱敏，用于前端展示"""
    result = {}
    for name, value in cookies.items():
        if len(value) > show_chars * 2:
            result[name] = value[:show_chars] + "***" + value[-show_chars:]
        else:
            result[name] = value[:2] + "***" if value else ""
    return result


def get_cookie_status() -> dict:
    """获取 Cookie 文件状态"""
    if not COOKIE_FILE.exists():
        return {"configured": False, "count": 0, "updated_at": None, "cookies": {}}

    mtime = datetime.fromtimestamp(COOKIE_FILE.stat().st_mtime)
    cookies = load_cookies()
    return {
        "configured": len(cookies) > 0,
        "count": len(cookies),
        "updated_at": mtime.isoformat(),
        "cookies": mask_cookies(cookies),
    }


def clear_cookies():
    """清除 Cookie 文件"""
    if COOKIE_FILE.exists():
        COOKIE_FILE.unlink()
        logger.info("✓ Cookie 已清除")


def _is_denied(name: str) -> bool:
    """判断是否属于追踪/统计 cookie，不需要注入"""
    for prefix in DENY_PREFIXES:
        if name.startswith(prefix):
            return True
    return False


def build_cookie_string(cookies: dict) -> str:
    """把 dict 拼成 Cookie header 字符串"""
    return "; ".join(f"{k}={v}" for k, v in cookies.items())
