"""
大模型客户端（基于 httpx，无需官方 SDK），支持两种接口协议：
- openai：/chat/completions（OpenAI 及各类兼容网关、vLLM、Ollama）
- anthropic：Claude Messages API /v1/messages（system 独立、max_tokens 必填、
  响应在 content[] 分块里，且不支持 response_format，JSON 约束改由提示词承担）

重试策略（共 3 次尝试，间隔 3s / 6s）：
- 传输层：网络错误 / 超时 / HTTP 408、409、429、5xx 自动重试，尊重 Retry-After
- 内容层：chat_json 在模型返回无法解析的 JSON 时原样重发重试
- 400（json_mode 不兼容）自动降级普通请求；401/403 等参数/鉴权错误立即失败
"""
import asyncio
import json
import logging
import re

import httpx

from app.storage import llm_config

logger = logging.getLogger(__name__)

# 最多尝试 3 次（首次 + 2 次重试），退避间隔与爬虫侧 3s/6s 保持一致
MAX_ATTEMPTS = 3
RETRY_DELAYS = (3, 6)
# 值得重试的 HTTP 状态码
# 500-504 服务端故障；520-524、527 为 Cloudflare 网关的源服务器错误/超时（
# 返回 HTML 错误页，属瞬时故障，重试通常可恢复），都按可重试处理
RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504, 520, 521, 522, 523, 524, 527}

# Claude Messages API 版本与输出上限（清洗长文需要较大额度，必填参数）
ANTHROPIC_VERSION = "2023-06-01"
ANTHROPIC_MAX_TOKENS = 8192

# 要求模型只输出 JSON 的补充约束（Claude 无 response_format，只能靠提示词）
JSON_ONLY_HINT = (
    "\n\n【输出格式】只输出一个合法的 JSON 对象，"
    "不要包含 markdown 代码围栏、注释或任何解释性文字。"
)


class LLMError(RuntimeError):
    pass


def is_configured() -> bool:
    cfg = llm_config.load_config()
    return bool(cfg["base_url"] and cfg["api_key"] and cfg["model"])


def extract_json(text: str):
    """从模型输出中容错提取 JSON 对象（去 markdown 围栏 / 截取首个 {...}）"""
    if not text:
        return None
    t = text.strip()
    t = re.sub(r"^```(?:json)?|```$", "", t, flags=re.MULTILINE).strip()
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", t, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    return None


def _error_detail(resp: "httpx.Response") -> str:
    """错误响应摘要：网关返回 HTML 错误页（如 Cloudflare 524 源超时）时不贴源码，
    改为可读提示；其余情况截取响应体前 200 字符"""
    body = resp.text or ""
    head = body.lstrip()[:64].lower()
    if head.startswith(("<html", "<!doctype")):
        return "网关错误页（源服务器超时或不可达，如 Cloudflare 524）"
    return body[:200]


def _retry_delay(attempt: int, resp: "httpx.Response | None" = None) -> float:
    """第 attempt 次失败后的等待秒数（attempt 从 1 开始）；优先采用服务端 Retry-After"""
    if resp is not None:
        ra = resp.headers.get("Retry-After")
        if ra:
            try:
                return max(0.0, min(float(ra), 30.0))
            except ValueError:
                pass
    idx = attempt - 1
    return RETRY_DELAYS[idx] if idx < len(RETRY_DELAYS) else 8.0


# ============ 协议适配 ============

def _messages_url(base_url: str) -> str:
    """
    拼出 Claude Messages 端点，兼容三种写法：
    https://api.anthropic.com        → .../v1/messages
    https://api.anthropic.com/v1     → .../v1/messages
    网关带前缀 .../anthropic/v1      → .../anthropic/v1/messages
    """
    b = base_url.rstrip("/")
    if b.endswith("/messages"):
        return b
    if b.endswith("/v1") or "/v1/" in b or re.search(r"/v\d+$", b):
        return b + "/messages"
    return b + "/v1/messages"


def _build_request(cfg: dict, system_prompt: str, user_prompt: str,
                   json_mode: bool) -> tuple[str, dict, dict]:
    """按渠道协议构造 (url, headers, payload)"""
    protocol = (cfg.get("protocol") or llm_config.DEFAULT_PROTOCOL).lower()

    if protocol == "anthropic":
        # Claude：system 独立传；messages 只放 user；max_tokens 必填
        sp = system_prompt + (JSON_ONLY_HINT if json_mode else "")
        return (
            _messages_url(cfg["base_url"]),
            {
                "x-api-key": cfg["api_key"],
                "anthropic-version": ANTHROPIC_VERSION,
                "Content-Type": "application/json",
            },
            {
                "model": cfg["model"],
                "system": sp,
                "messages": [{"role": "user", "content": user_prompt}],
                "max_tokens": ANTHROPIC_MAX_TOKENS,
                "temperature": cfg.get("temperature", 0.3),
            },
        )

    # 默认 OpenAI 兼容
    payload = {
        "model": cfg["model"],
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": cfg.get("temperature", 0.3),
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    return (
        cfg["base_url"].rstrip("/") + "/chat/completions",
        {
            "Authorization": f"Bearer {cfg['api_key']}",
            "Content-Type": "application/json",
        },
        payload,
    )


def _parse_reply(protocol: str, data: dict) -> str:
    """从两种协议的响应里取出助手文本；结构不符则抛异常触发重试"""
    if protocol == "anthropic":
        # content 是分块数组，可能先给 thinking 块，只拼接 text 块
        blocks = data.get("content")
        if not isinstance(blocks, list):
            raise KeyError("content")
        text = "".join(
            b.get("text", "") for b in blocks
            if isinstance(b, dict) and b.get("type") == "text"
        )
        if not text:
            raise KeyError("content[].text")
        return text
    return data["choices"][0]["message"]["content"]


async def chat(system_prompt: str, user_prompt: str, *, timeout: float = 300.0,
               json_mode: bool = True, cfg: dict | None = None) -> str:
    """
    按渠道协议调用对话接口，返回助手消息文本。
    cfg 为 None 时使用当前激活渠道；传入指定渠道配置可测试/调用非激活渠道。
    json_mode=True 时要求模型输出 JSON（OpenAI 走 response_format，Claude 走提示词约束）。
    网络错误 / 429 / 5xx 等瞬时故障按 RETRY_DELAYS 指数退避重试。
    """
    cfg = cfg if cfg is not None else llm_config.load_config()
    if not (cfg["base_url"] and cfg["api_key"] and cfg["model"]):
        raise LLMError("大模型未配置：请先在「数据清洗」页填写 base_url / API Key / 模型")

    protocol = (cfg.get("protocol") or llm_config.DEFAULT_PROTOCOL).lower()

    async def _post(use_json_mode: bool) -> httpx.Response:
        url, headers, payload = _build_request(
            cfg, system_prompt, user_prompt, use_json_mode
        )
        async with httpx.AsyncClient(timeout=timeout) as client:
            return await client.post(url, headers=headers, json=payload)

    last_error: Exception | None = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        resp = None
        try:
            resp = await _post(json_mode)
            # 个别兼容服务不认 response_format，降级为普通请求（不当作可重试故障）
            if (resp.status_code == 400 and json_mode
                    and "response_format" in resp.text):
                resp = await _post(False)

            if resp.status_code in RETRYABLE_STATUS and attempt < MAX_ATTEMPTS:
                delay = _retry_delay(attempt, resp)
                logger.warning(
                    "大模型返回 HTTP %s，第 %s/%s 次将在 %.0fs 后重试",
                    resp.status_code, attempt, MAX_ATTEMPTS, delay,
                )
                await asyncio.sleep(delay)
                continue

            if resp.status_code != 200:
                raise LLMError(f"大模型返回 HTTP {resp.status_code}：{_error_detail(resp)}")

            try:
                return _parse_reply(protocol, resp.json())
            except (json.JSONDecodeError, KeyError, IndexError, TypeError) as e:
                # 200 但体不是预期结构（常见于网关/代理返回 HTML 错误页）：重试
                if attempt >= MAX_ATTEMPTS:
                    raise LLMError(f"大模型响应结构异常（已重试 {MAX_ATTEMPTS - 1} 次）：{e}")
                delay = _retry_delay(attempt)
                logger.warning(
                    "大模型响应结构异常：%s，第 %s/%s 次将在 %.0fs 后重试",
                    e, attempt, MAX_ATTEMPTS, delay,
                )
                await asyncio.sleep(delay)

        except httpx.HTTPError as e:
            last_error = e
            if attempt >= MAX_ATTEMPTS:
                raise LLMError(f"请求大模型失败（已重试 {MAX_ATTEMPTS - 1} 次）：{str(e)[:150]}")
            delay = _retry_delay(attempt)
            logger.warning(
                "请求大模型网络错误：%s，第 %s/%s 次将在 %.0fs 后重试",
                str(e)[:100], attempt, MAX_ATTEMPTS, delay,
            )
            await asyncio.sleep(delay)

    # 理论上不会走到这里
    raise LLMError(f"请求大模型失败：{last_error}")


async def chat_json(system_prompt: str, user_prompt: str, *, timeout: float = 300.0,
                    max_attempts: int = MAX_ATTEMPTS, cfg: dict | None = None):
    """
    要求模型返回 JSON 的对话：自动解析并容错重试。
    模型偶尔会夹带解释/markdown 导致解析失败，此时原样重发，最多 max_attempts 次。
    """
    last_raw = ""
    for attempt in range(1, max_attempts + 1):
        raw = await chat(system_prompt, user_prompt, timeout=timeout,
                         json_mode=True, cfg=cfg)
        last_raw = raw
        data = extract_json(raw)
        if data is not None:
            return data
        if attempt < max_attempts:
            delay = _retry_delay(attempt)
            logger.warning(
                "模型输出不是合法 JSON（第 %s/%s 次），%.0fs 后重试；返回片段：%s",
                attempt, max_attempts, delay, (raw or "")[:120],
            )
            await asyncio.sleep(delay)

    raise LLMError(
        f"模型连续 {max_attempts} 次未返回合法 JSON，请稍后重试或更换模型；最后返回片段：{last_raw[:150]}"
    )


async def ping(cfg: dict | None = None) -> dict:
    """连通性测试：发一条最小请求，返回模型回复；cfg 为 None 时测当前激活渠道"""
    reply = await chat(
        "你是连通性测试助手，只回复 pong 两个小写字母。",
        "ping", timeout=30.0, json_mode=False, cfg=cfg,
    )
    return {"reply": (reply or "").strip()[:50]}
