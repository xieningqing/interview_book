"""
IP 代理池管理
- 支持单个代理 / 代理列表 / 轮换策略
- 文件持久化 data/proxies.json
- httpx 客户端原生支持 "http://host:port" 格式
"""
import json
import random
import logging
from pathlib import Path
from typing import Optional
from datetime import datetime

from app.config import DATA_DIR

logger = logging.getLogger(__name__)

PROXY_FILE = DATA_DIR / "proxies.json"

# 轮换策略
STRATEGIES = ("round_robin", "random", "first")


def _default_config() -> dict:
    return {
        "enabled": False,
        "strategy": "round_robin",    # round_robin / random / first
        "current_index": 0,
        "proxies": [],               # ["http://ip:port", ...]
        "updated_at": None,
    }


def load_proxies() -> dict:
    """加载代理配置"""
    if not PROXY_FILE.exists():
        return _default_config()
    try:
        data = json.loads(PROXY_FILE.read_text(encoding="utf-8"))
        # 填充默认值确保向后兼容
        default = _default_config()
        for k, v in default.items():
            data.setdefault(k, v)
        if "current_index" not in data:
            data["current_index"] = 0
        return data
    except Exception as e:
        logger.warning(f"加载 proxies.json 失败: {e}")
        return _default_config()


def save_proxies(config: dict) -> dict:
    """保存代理配置"""
    PROXY_FILE.parent.mkdir(parents=True, exist_ok=True)
    config["updated_at"] = datetime.now().isoformat()
    PROXY_FILE.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    return config


def set_proxies(proxy_list: list[str], enabled: bool = True, strategy: str = "round_robin") -> dict:
    """设置代理列表（便捷方法）"""
    cfg = load_proxies()
    cfg["proxies"] = [p.strip() for p in proxy_list if p.strip()]
    cfg["enabled"] = enabled
    cfg["strategy"] = strategy if strategy in STRATEGIES else "round_robin"
    cfg["current_index"] = 0
    return save_proxies(cfg)


def add_proxy(proxy: str) -> dict:
    """追加单个代理"""
    cfg = load_proxies()
    proxy = proxy.strip()
    if proxy and proxy not in cfg["proxies"]:
        cfg["proxies"].append(proxy)
        save_proxies(cfg)
    return cfg


def remove_proxy(proxy: str) -> dict:
    """移除单个代理"""
    cfg = load_proxies()
    cfg["proxies"] = [p for p in cfg["proxies"] if p != proxy]
    return save_proxies(cfg)


def clear_proxies() -> dict:
    """清空代理"""
    cfg = load_proxies()
    cfg["proxies"] = []
    cfg["enabled"] = False
    return save_proxies(cfg)


def toggle_enabled(enabled: bool) -> dict:
    cfg = load_proxies()
    cfg["enabled"] = enabled
    return save_proxies(cfg)


def pick_proxy() -> Optional[str]:
    """
    按轮换策略选一个代理。返回代理 URL 或 None（未启用/无代理）。
    每次调用轮询索引自增，实现 round-robin。

    注意：每次调用都可能换到不同代理。单 cookie 场景下不要在每个请求上
    调用它（同一账号 IP 乱跳会触发风控）；应在一次爬取任务开始时调用一次，
    把返回的 URL 传给该任务的所有 httpx 客户端复用（会话粘性）。
    """
    cfg = load_proxies()
    if not cfg.get("enabled") or not cfg.get("proxies"):
        return None

    proxies = cfg["proxies"]
    strategy = cfg.get("strategy", "round_robin")

    if strategy == "random":
        proxy = random.choice(proxies)
    elif strategy == "first":
        proxy = proxies[0]
    else:  # round_robin
        idx = cfg.get("current_index", 0) % len(proxies)
        proxy = proxies[idx]
        cfg["current_index"] = (idx + 1) % len(proxies)
        save_proxies(cfg)

    return proxy


# 说明：httpx 0.26 起 proxies={"http://": ..., "https://": ...} 已弃用，
# 0.28 起直接移除；统一改用 AsyncClient(proxy="<url>")。
# 因此本模块只返回代理 URL 字符串（pick_proxy），不再提供 dict 转换。


def get_proxy_status() -> dict:
    """获取代理状态（不含敏感值）"""
    cfg = load_proxies()
    proxies = cfg.get("proxies", [])
    # 脱敏
    masked = [_mask_proxy(p) for p in proxies]
    return {
        "enabled": cfg.get("enabled", False),
        "strategy": cfg.get("strategy", "round_robin"),
        "count": len(proxies),
        "proxies": masked,
        "updated_at": cfg.get("updated_at"),
    }


def _mask_proxy(proxy: Optional[str]) -> str:
    """脱敏代理地址：只显示主机和端口，隐藏用户名密码（兼容 http/https/socks5）"""
    if not proxy:
        return ""
    # scheme://user:pass@host:port → scheme://***@host:port
    import re
    m = re.match(r"([a-zA-Z][a-zA-Z0-9+.-]*://)([^@]+@)?(.+)", proxy)
    if m:
        scheme, creds, host = m.groups()
        if creds:
            return f"{scheme}***@{host}"
        return proxy
    return proxy
