"""
OpenAI 兼容 Chat Completions 客户端（基于 httpx，无需 openai SDK）
"""
import json
import logging

import httpx

from app.storage import llm_config

logger = logging.getLogger(__name__)


class LLMError(RuntimeError):
    pass


def is_configured() -> bool:
    cfg = llm_config.load_config()
    return bool(cfg["base_url"] and cfg["api_key"] and cfg["model"])


async def chat(system_prompt: str, user_prompt: str, *, timeout: float = 90.0,
               json_mode: bool = True) -> str:
    """
    调用 chat/completions，返回助手消息文本。
    json_mode=True 时要求模型输出 JSON（部分兼容服务不支持该参数则自动降级重试）。
    """
    cfg = llm_config.load_config()
    if not (cfg["base_url"] and cfg["api_key"] and cfg["model"]):
        raise LLMError("大模型未配置：请先在「数据清洗」页填写 base_url / API Key / 模型")

    url = cfg["base_url"].rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {cfg['api_key']}",
        "Content-Type": "application/json",
    }
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    async def _post(use_json_mode: bool) -> httpx.Response:
        payload = {
            "model": cfg["model"],
            "messages": messages,
            "temperature": cfg.get("temperature", 0.3),
        }
        if use_json_mode:
            payload["response_format"] = {"type": "json_object"}
        async with httpx.AsyncClient(timeout=timeout) as client:
            return await client.post(url, headers=headers, json=payload)

    try:
        resp = await _post(json_mode)
        # 个别兼容服务不认 response_format，降级为普通请求
        if resp.status_code == 400 and json_mode and "response_format" in resp.text:
            resp = await _post(False)
    except httpx.HTTPError as e:
        raise LLMError(f"请求大模型失败：{str(e)[:150]}")

    if resp.status_code != 200:
        raise LLMError(f"大模型返回 HTTP {resp.status_code}：{resp.text[:200]}")

    try:
        data = resp.json()
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, json.JSONDecodeError) as e:
        raise LLMError(f"大模型响应结构异常：{e}")


async def ping() -> dict:
    """连通性测试：发一条最小请求，返回模型回复"""
    reply = await chat(
        "你是连通性测试助手，只回复 pong 两个小写字母。",
        "ping", timeout=30.0, json_mode=False,
    )
    return {"reply": (reply or "").strip()[:50]}
