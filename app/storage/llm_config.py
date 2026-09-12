"""
大模型接口配置（OpenAI 兼容协议）
- 本地文件持久化 data/llm_config.json（含 API Key，绝不提交到仓库）
- DeepSeek / 通义千问 / Kimi / 智谱 / 自建 vLLM 等均兼容
"""
import json
import logging
from datetime import datetime

from app.config import DATA_DIR

logger = logging.getLogger(__name__)

CONFIG_FILE = DATA_DIR / "llm_config.json"


# 结构化抽取/清洗任务用固定低温度，保证输出稳定、可解析（不对用户暴露）
FIXED_TEMPERATURE = 0.3


def _default_config() -> dict:
    return {
        "base_url": "",          # 如 https://api.deepseek.com/v1
        "api_key": "",
        "model": "",             # 如 deepseek-chat / qwen-plus / gpt-4o-mini
        "temperature": FIXED_TEMPERATURE,
        "updated_at": None,
    }


def load_config() -> dict:
    """加载配置（含明文 key，仅供服务端调用使用）"""
    if not CONFIG_FILE.exists():
        return _default_config()
    try:
        data = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
        cfg = _default_config()
        cfg.update({k: v for k, v in data.items() if k in cfg})
        return cfg
    except Exception as e:
        logger.warning(f"加载 llm_config.json 失败: {e}")
        return _default_config()


def save_config(base_url: str, api_key: str, model: str) -> dict:
    """保存配置；api_key 传空串表示沿用原值（编辑时不回显明文）。温度固定不对用户暴露"""
    cfg = load_config()
    base_url = (base_url or "").strip().rstrip("/")
    model = (model or "").strip()
    if api_key and api_key.strip():
        cfg["api_key"] = api_key.strip()
    cfg["base_url"] = base_url
    cfg["model"] = model
    cfg["temperature"] = FIXED_TEMPERATURE
    cfg["updated_at"] = datetime.now().isoformat()

    CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    return cfg


def get_status() -> dict:
    """前端展示用状态：key 只显示是否已配置/前后几位，不回传明文"""
    cfg = load_config()
    key = cfg.get("api_key", "")
    masked = ""
    if key:
        masked = key[:4] + "***" + key[-4:] if len(key) > 10 else "***"
    return {
        "configured": bool(cfg["base_url"] and key and cfg["model"]),
        "base_url": cfg["base_url"],
        "model": cfg["model"],
        "api_key_masked": masked,
        "updated_at": cfg.get("updated_at"),
    }
