"""
大模型接口配置（支持 OpenAI 兼容 / Anthropic Claude 两种协议）
- 本地文件持久化 data/llm_config.json（含 API Key，绝不提交到仓库）
- 支持保存多个渠道（DeepSeek / 通义千问 / Kimi / Claude / 自建网关等），一键切换当前渠道
- 兼容旧版单配置文件：首次加载自动升级为多渠道结构
"""
import json
import logging
import uuid
from datetime import datetime

from app.config import DATA_DIR

logger = logging.getLogger(__name__)

CONFIG_FILE = DATA_DIR / "llm_config.json"

# 结构化抽取/清洗任务用固定低温度，保证输出稳定、可解析（不对用户暴露）
FIXED_TEMPERATURE = 0.3

# 接口协议：openai = /chat/completions；anthropic = Claude Messages API(/v1/messages)
PROTOCOLS = ("openai", "anthropic")
DEFAULT_PROTOCOL = "openai"


def _empty_flat_config() -> dict:
    """当前渠道的扁平结构（llm.chat 使用）"""
    return {
        "base_url": "",
        "api_key": "",
        "model": "",
        "protocol": DEFAULT_PROTOCOL,
        "temperature": FIXED_TEMPERATURE,
        "updated_at": None,
    }


def _default_store() -> dict:
    return {"active_id": None, "configs": []}


def _normalize_item(item: dict) -> dict:
    protocol = (item.get("protocol") or DEFAULT_PROTOCOL).strip().lower()
    if protocol not in PROTOCOLS:
        protocol = DEFAULT_PROTOCOL
    return {
        "id": item.get("id") or f"chan_{uuid.uuid4().hex[:12]}",
        "name": (item.get("name") or "").strip() or (item.get("model") or "未命名渠道"),
        "base_url": (item.get("base_url") or "").strip().rstrip("/"),
        "api_key": item.get("api_key") or "",
        "model": (item.get("model") or "").strip(),
        "protocol": protocol,
        "temperature": item.get("temperature", FIXED_TEMPERATURE),
        "updated_at": item.get("updated_at"),
    }


def _write_store(store: dict):
    CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE.write_text(
        json.dumps(store, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def load_store() -> dict:
    """加载完整多渠道配置；旧版单配置自动迁移并回写"""
    if not CONFIG_FILE.exists():
        return _default_store()
    try:
        data = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning(f"加载 llm_config.json 失败: {e}")
        return _default_store()

    # 新格式
    if isinstance(data, dict) and isinstance(data.get("configs"), list):
        store = _default_store()
        store["configs"] = [_normalize_item(x) for x in data["configs"]]
        active_id = data.get("active_id")
        ids = {c["id"] for c in store["configs"]}
        store["active_id"] = active_id if active_id in ids else (
            store["configs"][0]["id"] if store["configs"] else None
        )
        return store

    # 旧版单配置迁移：顶层有 base_url/api_key/model
    if isinstance(data, dict) and (data.get("base_url") or data.get("model")):
        item = _normalize_item({**data, "id": "chan_default", "name": "默认渠道"})
        store = {"active_id": item["id"], "configs": [item]}
        try:
            _write_store(store)
            logger.info("已将旧版单模型配置迁移为多渠道配置")
        except Exception as e:
            logger.warning(f"迁移回写 llm_config.json 失败: {e}")
        return store

    return _default_store()


def load_config() -> dict:
    """当前激活渠道的扁平配置（含明文 key，仅供服务端调用使用）；无渠道时返回空配置"""
    store = load_store()
    for c in store["configs"]:
        if c["id"] == store["active_id"]:
            cfg = _empty_flat_config()
            cfg.update({
                "base_url": c["base_url"],
                "api_key": c["api_key"],
                "model": c["model"],
                "protocol": c.get("protocol", DEFAULT_PROTOCOL),
                "temperature": c.get("temperature", FIXED_TEMPERATURE),
                "updated_at": c.get("updated_at"),
            })
            return cfg
    return _empty_flat_config()


def get_config_by_id(channel_id: str) -> dict | None:
    """按 id 取渠道明文配置（仅服务端测试连接用）"""
    store = load_store()
    for c in store["configs"]:
        if c["id"] == channel_id:
            return c
    return None


def _mask(key: str) -> str:
    if not key:
        return ""
    return key[:4] + "***" + key[-4:] if len(key) > 10 else "***"


def list_status() -> dict:
    """前端展示用：渠道列表脱敏 + 当前激活 id"""
    store = load_store()
    channels = []
    for c in store["configs"]:
        channels.append({
            "id": c["id"],
            "name": c["name"],
            "base_url": c["base_url"],
            "model": c["model"],
            "protocol": c.get("protocol", DEFAULT_PROTOCOL),
            "has_key": bool(c["api_key"]),
            "api_key_masked": _mask(c["api_key"]),
            "updated_at": c.get("updated_at"),
        })
    active = next(
        (c for c in channels if c["id"] == store["active_id"]), None
    )
    configured = bool(
        active and active["base_url"] and active["has_key"] and active["model"]
    )
    return {
        "configured": configured,
        "active_id": store["active_id"],
        "channels": channels,
        # 兼容旧模板头部字段
        "base_url": active["base_url"] if active else "",
        "model": active["model"] if active else "",
        "protocol": active["protocol"] if active else DEFAULT_PROTOCOL,
        "api_key_masked": active["api_key_masked"] if active else "",
        "updated_at": active["updated_at"] if active else None,
    }


# 旧调用点兼容别名
get_status = list_status


def upsert_config(
    base_url: str,
    model: str,
    api_key: str = "",
    *,
    channel_id: str | None = None,
    name: str | None = None,
    protocol: str | None = None,
    activate: bool = False,
) -> dict:
    """
    新增或更新渠道。
    - channel_id 为空 → 新增；非空 → 更新（api_key 留空沿用原值）
    - protocol 为 openai / anthropic，决定调用哪种接口协议
    - activate=True 或当前没有激活渠道时，保存后自动切为当前渠道
    """
    store = load_store()
    base_url = (base_url or "").strip().rstrip("/")
    model = (model or "").strip()
    now = datetime.now().isoformat()

    item = None
    if channel_id:
        item = next((c for c in store["configs"] if c["id"] == channel_id), None)
    if item is None:
        item = _normalize_item({"id": f"chan_{uuid.uuid4().hex[:12]}"})
        store["configs"].append(item)

    item["base_url"] = base_url
    item["model"] = model
    if protocol:
        p = protocol.strip().lower()
        if p not in PROTOCOLS:
            raise ValueError(f"不支持的协议：{protocol}")
        item["protocol"] = p
    if api_key and api_key.strip():
        item["api_key"] = api_key.strip()
    if name and name.strip():
        item["name"] = name.strip()
    elif not item.get("name") or item["name"] == item.get("model"):
        item["name"] = model or item["name"]
    item["temperature"] = FIXED_TEMPERATURE
    item["updated_at"] = now

    if activate or not store["active_id"]:
        store["active_id"] = item["id"]

    _write_store(store)
    return list_status()


def activate_config(channel_id: str) -> dict:
    """切换当前使用的渠道"""
    store = load_store()
    if not any(c["id"] == channel_id for c in store["configs"]):
        raise KeyError(channel_id)
    store["active_id"] = channel_id
    _write_store(store)
    return list_status()


def delete_config(channel_id: str) -> dict:
    """删除渠道；删的是当前渠道时自动切到剩余第一个"""
    store = load_store()
    before = len(store["configs"])
    store["configs"] = [c for c in store["configs"] if c["id"] != channel_id]
    if len(store["configs"]) == before:
        raise KeyError(channel_id)
    if store["active_id"] == channel_id:
        store["active_id"] = store["configs"][0]["id"] if store["configs"] else None
    _write_store(store)
    return list_status()


def save_config(base_url: str, api_key: str, model: str) -> dict:
    """旧接口兼容：更新当前激活渠道（没有则新建）"""
    store = load_store()
    active_id = store["active_id"]
    if not active_id and store["configs"]:
        active_id = store["configs"][0]["id"]
    return upsert_config(
        base_url, model, api_key, channel_id=active_id, activate=True
    )
