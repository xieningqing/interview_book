"""
牛客网面经爬虫
- 关键词搜索（支持按时间范围过滤）
- 抓取详情页完整内容
- 返回结构化数据
- 反爬：UA 轮换、随机延迟、重试退避、风控检测

进度回调格式:
  progress_callback({
      "step": "search_page" | "fetch_detail" | "done" | "risk_warning",
      "phase": "search" | "detail" | "complete",     # 大阶段
      "status": "success" | "failed" | "skipped" | "risk",
      "message": "...",
      "current": 0,
      "total": 0,
      "extra": {...}  # 附加数据（耗时、状态码、标题等）
  })
"""
import asyncio
import re
import time
import random
import logging
from contextvars import ContextVar
from datetime import datetime, timedelta
from typing import Optional

import httpx
from app.config import CRAWLER, UA_POOL, BROWSER_HEADERS_TEMPLATE
from app.storage.cookie_manager import load_cookies
from app.storage.proxy_manager import pick_proxy, _mask_proxy

logger = logging.getLogger(__name__)

# ============ Cookie 缓存 ============

_COOKIE_CACHE: dict | None = None


def _active_cookies() -> dict:
    global _COOKIE_CACHE
    if _COOKIE_CACHE is None:
        _COOKIE_CACHE = load_cookies()
    return _COOKIE_CACHE


def invalidate_cookie_cache():
    global _COOKIE_CACHE
    _COOKIE_CACHE = None


# ============ 反爬工具函数 ============

# 任务级 UA：一次爬取内固定，与锁定的代理 IP 保持指纹一致
_TASK_UA: ContextVar[Optional[str]] = ContextVar("_TASK_UA", default=None)


def _random_ua() -> str:
    """从 UA 池随机选一个"""
    return random.choice(UA_POOL)


def _random_delay(base: float, jitter: float) -> float:
    """基础延迟 + 随机抖动"""
    return base + random.uniform(0, jitter)


async def _smart_sleep(base: float, jitter: float = 0.0):
    """智能 sleep，带随机抖动"""
    actual = _random_delay(base, jitter) if jitter else base
    await asyncio.sleep(actual)


def _build_headers(content_type: bool = True) -> dict:
    """构建更像真实浏览器的请求头

    同一爬取任务内 UA 固定（见 _TASK_UA），且 Sec-Ch-Ua* 客户端提示与 UA
    严格对应——Firefox 不发这些头，Chrome/Edge 的版本号要与 UA 一致。
    """
    headers = dict(BROWSER_HEADERS_TEMPLATE)
    ua = _TASK_UA.get() or _random_ua()
    headers["User-Agent"] = ua
    if content_type:
        headers["Content-Type"] = "application/json; charset=UTF-8"

    if "Firefox" in ua:
        # Firefox 不发送 Sec-Ch-Ua* 系列
        headers.pop("Sec-Ch-Ua", None)
        headers.pop("Sec-Ch-Ua-Mobile", None)
        headers.pop("Sec-Ch-Ua-Platform", None)
    else:
        # 让客户端提示与 UA 的浏览器/平台/版本对齐
        m = re.search(r"Chrome/(\d+)", ua)
        if m:
            v = m.group(1)
            brand = "Microsoft Edge" if "Edg/" in ua else "Google Chrome"
            headers["Sec-Ch-Ua"] = (
                f'"Not_A Brand";v="8", "Chromium";v="{v}", "{brand}";v="{v}"'
            )
        headers["Sec-Ch-Ua-Platform"] = '"macOS"' if "Macintosh" in ua else '"Windows"'
    return headers


def _is_risk_response(status_code: int, text: str) -> Optional[str]:
    """检测响应是否命中风控，返回命中的关键词或 None"""
    if status_code in (403, 429, 503):
        return f"HTTP {status_code}"
    text_lower = text[:2000].lower()
    for kw in CRAWLER["risk_keywords"]:
        if kw.lower() in text_lower:
            return kw
    return None


# ============ 带重试的 HTTP 客户端 ============

async def _post_with_retry(
    client: httpx.AsyncClient,
    url: str,
    json_payload: dict,
    headers: dict,
    progress_cb=None,
    label: str = "",
) -> tuple[Optional[httpx.Response], str, float]:
    """
    POST 请求 + 自动重试 + 风控检测

    Returns:
        (response_or_None, status_label, elapsed_seconds)
        status_label: "success" / "failed" / "risk" / "timeout"
    """
    max_retries = CRAWLER["max_retries"]
    base_delay = CRAWLER["retry_base_delay"]
    start = time.monotonic()

    for attempt in range(max_retries + 1):
        try:
            resp = await client.post(url, json=json_payload, headers=headers)
            elapsed = time.monotonic() - start

            # 风控检测
            risk_hit = _is_risk_response(resp.status_code, resp.text)
            if risk_hit:
                label_out = "risk"
                msg = f"🚨 风控命中 [{label}]: {risk_hit} (HTTP {resp.status_code}, 第{attempt+1}次)"
                if progress_cb:
                    progress_cb({
                        "step": f"risk_{label}", "phase": "search", "status": "risk",
                        "message": msg, "current": 0, "total": 0,
                        "extra": {"attempt": attempt + 1, "http_code": resp.status_code, "risk_keyword": risk_hit},
                    })
                # 风控就不等重试了，直接放弃
                return None, "risk", elapsed

            if resp.status_code == 200:
                return resp, "success", elapsed

            # 非 200 非风控，尝试重试
            if attempt < max_retries:
                wait = base_delay * (2 ** attempt) + random.uniform(0, 1)
                logger.warning(f"[{label}] HTTP {resp.status_code}, {wait:.1f}s 后重试 ({attempt+1}/{max_retries})")
                await asyncio.sleep(wait)

        except httpx.TimeoutException:
            elapsed = time.monotonic() - start
            label_out = "timeout"
            if attempt < max_retries:
                wait = base_delay * (2 ** attempt)
                logger.warning(f"[{label}] 超时, {wait:.1f}s 后重试 ({attempt+1}/{max_retries})")
                await asyncio.sleep(wait)
            else:
                if progress_cb:
                    progress_cb({
                        "step": f"timeout_{label}", "phase": "search", "status": "failed",
                        "message": f"⚠️ [{label}] 请求超时（已重试{max_retries}次）",
                        "current": 0, "total": 0,
                    })

        except Exception as e:
            elapsed = time.monotonic() - start
            logger.warning(f"[{label}] 请求异常: {e}")
            if attempt < max_retries:
                await asyncio.sleep(base_delay * (2 ** attempt))

    return None, "failed", time.monotonic() - start


async def _get_with_retry(
    client: httpx.AsyncClient,
    url: str,
    headers: dict,
    progress_cb=None,
    label: str = "",
) -> tuple[Optional[httpx.Response], str, float]:
    """GET 请求 + 自动重试 + 风控检测"""
    max_retries = CRAWLER["max_retries"]
    base_delay = CRAWLER["retry_base_delay"]
    start = time.monotonic()

    for attempt in range(max_retries + 1):
        try:
            resp = await client.get(url, headers=headers)
            elapsed = time.monotonic() - start

            risk_hit = _is_risk_response(resp.status_code, resp.text)
            if risk_hit:
                if progress_cb:
                    progress_cb({
                        "step": f"risk_{label}", "phase": "detail", "status": "risk",
                        "message": f"🚨 风控 [{label}]: {risk_hit}",
                        "current": 0, "total": 0,
                        "extra": {"risk_keyword": risk_hit},
                    })
                return None, "risk", elapsed

            if resp.status_code == 200:
                return resp, "success", elapsed

            if attempt < max_retries:
                wait = base_delay * (2 ** attempt) + random.uniform(0, 1)
                await asyncio.sleep(wait)

        except httpx.TimeoutException:
            if attempt < max_retries:
                await asyncio.sleep(base_delay * (2 ** attempt))
        except Exception as e:
            logger.warning(f"[{label}] GET 异常: {e}")

    return None, "failed", time.monotonic() - start


# ============ HTML 处理 ============

def html_to_text(html: str) -> str:
    if not html:
        return ""
    html = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r"<style[^>]*>.*?</style>", "", html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r"</?(p|div|br|h[1-6]|li|tr|blockquote|ul|ol)[^>]*>", "\n", html, flags=re.IGNORECASE)
    html = re.sub(r"<[^>]+>", "", html)
    for old, new in [
        ("&nbsp;", " "), ("&amp;", "&"), ("&lt;", "<"),
        ("&gt;", ">"), ("&quot;", '"'), ("&#39;", "'"), ("\xa0", " ")
    ]:
        html = html.replace(old, new)
    html = re.sub(r"\n\s*\n", "\n\n", html)
    return html.strip()


def parse_timestamp(ts) -> Optional[datetime]:
    if ts is None:
        return None
    if isinstance(ts, (int, float)) and ts > 1000000000000:  # 毫秒
        try: return datetime.fromtimestamp(ts / 1000)
        except (ValueError, OSError): pass
    if isinstance(ts, (int, float)) and ts > 1000000000:  # 秒
        try: return datetime.fromtimestamp(ts)
        except (ValueError, OSError): pass
    if isinstance(ts, str):
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%Y/%m/%d %H:%M:%S", "%Y/%m/%d"):
            try: return datetime.strptime(ts.strip(), fmt)
            except ValueError: continue
    return None


# ============ 核心爬取逻辑 ============

def _parse_entries(data: dict, query: str, seen: set, cutoff_time: Optional[datetime]) -> list[dict]:
    """解析单页搜索结果为候选条目；seen 去重、cutoff_time 时间过滤"""
    entries = []
    for r in data.get("data", {}).get("records", []):
        rc_type = r.get("rc_type", 0)
        rd = r.get("data", {})

        publish_time = None
        if rc_type == 201:
            md = rd.get("momentData", {})
            ts = md.get("createTime") or md.get("publishTime") or md.get("time")
            publish_time = parse_timestamp(ts)
        elif rc_type == 207:
            cd = rd.get("contentData", {})
            ts = cd.get("createTime") or cd.get("publishTime") or cd.get("updateTime")
            publish_time = parse_timestamp(ts)

        if cutoff_time and publish_time and publish_time < cutoff_time:
            continue

        entry = None
        if rc_type == 201:
            md = rd.get("momentData", {})
            uid = md.get("uuid", "")
            if uid and uid not in seen:
                seen.add(uid)
                tags = [t.get("name", "") for t in (md.get("tags") or [])]
                entry = {
                    "rc_type": 201, "uuid": uid,
                    "title": md.get("title", ""),
                    "author": md.get("user", {}).get("nickname", ""),
                    "school": md.get("school", ""),
                    "position": md.get("job", "") or md.get("position", ""),
                    "tags": tags,
                    "publish_time": publish_time,
                    "query": query,
                }
        elif rc_type == 207:
            cd = rd.get("contentData", {})
            cid = str(cd.get("id", ""))
            if cid and cid not in seen:
                seen.add(cid)
                tags = [t.get("name", "") for t in (cd.get("tags") or [])]
                entry = {
                    "rc_type": 207, "content_id": cid,
                    "title": cd.get("title", ""),
                    "author": cd.get("user", {}).get("nickname", ""),
                    "school": cd.get("school", ""),
                    "position": cd.get("job", "") or cd.get("position", ""),
                    "tags": tags,
                    "publish_time": publish_time,
                    "query": query,
                }

        if entry:
            entries.append(entry)
    return entries


async def iter_search_entries(
    query: str,
    max_pages: int = 5,
    days_filter: Optional[int] = None,
    progress_cb=None,
    proxy: Optional[str] = None,
):
    """
    按页搜索面经候选，每页 yield 一次候选条目列表（async generator）。

    调用方应边收边抓详情：正文过短/失败的条目不占目标名额，继续取下一页
    补充，直到拿够目标条数或翻完 max_pages 页。

    proxy: 本次任务锁定的代理 URL（会话粘性）。None 表示直连；
           不要在每个请求上单独轮换，单 cookie + IP 乱跳最容易触发风控。
    """
    seen = set()
    # 按自然日过滤：今天往前数 N 天的 0 点起（含今天共 N 天），而非精确到小时的滚动窗口
    if days_filter:
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        cutoff_time = today - timedelta(days=days_filter - 1)
    else:
        cutoff_time = None

    client = httpx.AsyncClient(
        timeout=CRAWLER["timeout"] + 5,
        follow_redirects=True,
        cookies=_active_cookies(),
        proxy=proxy,
    )
    try:
        for page in range(1, max_pages + 1):
            headers = _build_headers(content_type=True)
            payload = {
                "type": "all",
                "query": query,
                "page": page,
                "tag": [{"name": "面经", "id": 818, "count": None}],
                "order": "create",
                "gioParams": {"searchFrom_var": "顶部导航栏", "searchEnter_var": "主站"},
            }

            resp, status, elapsed = await _post_with_retry(
                client, CRAWLER["search_api"], payload, headers,
                progress_cb=progress_cb, label=f"search_page_{page}",
            )

            if progress_cb:
                progress_cb({
                    "step": "search_page", "phase": "search",
                    "status": status,
                    "message": f"搜索第 {page} 页 → {status.upper()} ({elapsed:.1f}s)",
                    "current": page, "total": max_pages,
                    "extra": {"elapsed": round(elapsed, 2), "page": page},
                })

            if resp is None:
                if status == "risk":
                    logger.warning("命中风控，停止搜索")
                return

            try:
                data = resp.json()
            except Exception:
                return

            if not data.get("success"):
                logger.warning(f"搜索返回失败: {data.get('message')}")
                return

            if not data.get("data", {}).get("records"):
                return

            entries = _parse_entries(data, query, seen, cutoff_time)
            if entries:
                yield entries

            total_page = data.get("data", {}).get("totalPage", 1)
            if page >= total_page:
                return

            # 搜索页间随机间隔
            await _smart_sleep(CRAWLER["delay"], CRAWLER["jitter"])
    finally:
        await client.aclose()


def _is_full_feed_page(html: str) -> bool:
    """
    判断是否为带正文的全量详情页。

    牛客在高频请求时会偶发返回一个 ~6KB 的 SEO 降级壳页：HTTP 200、标题
    正常，但正文只剩 <meta name="description"> 里的截断摘要，既没有
    feed-content-text 容器，也没有内嵌 JSON 的 content 字段。这种页面不能
    用来判断「正文过短」，必须退避后重试拿全量页。
    """
    return ("feed-content-text" in html) or ('"content":"' in html)


def _extract_feed(html: str) -> tuple[str, str]:
    """从全量详情页 HTML 提取 (title, content)；content 可能为空（真·空帖）"""
    title_m = re.search(r'"title":"([^"]+)"', html)
    title = title_m.group(1) if title_m else ""

    content = ""
    cm = re.search(
        r'<div[^>]*class="[^"]*feed-content-text[^"]*"[^>]*>(.*?)</div>',
        html, re.DOTALL | re.IGNORECASE
    )
    if cm:
        content = html_to_text(cm.group(1))
    if not content:
        for m in re.findall(r'"content":"([^"]{100,})"', html):
            content = m.replace("\\n", "\n").replace("\\u002F", "/").replace("\\t", "\t")
            break
    return title, content


async def _fetch_feed(client: httpx.AsyncClient, uuid: str, progress_cb=None) -> Optional[dict]:
    headers = _build_headers(content_type=False)
    url = f"{CRAWLER['feed_url']}/{uuid}"
    label = f"feed_{uuid[:12]}"
    max_retries = CRAWLER["max_retries"]

    # 外层重试：200 但拿到 SEO 降级壳页时，退避后重新请求全量页
    for attempt in range(max_retries + 1):
        resp, status, elapsed = await _get_with_retry(
            client, url, headers, progress_cb=progress_cb, label=label,
        )
        if resp is None:
            return None

        html = resp.text
        if "内容不存在" in html:
            return None

        title, content = _extract_feed(html)

        # 全量页：正文长短都采信（真的很短由上层判 skip_short）
        if content or _is_full_feed_page(html):
            return {"title": title, "content": content, "url": url}

        # 200 降级壳页：退避重试
        if attempt < max_retries:
            wait = CRAWLER["retry_base_delay"] * (2 ** attempt)
            if progress_cb:
                progress_cb({
                    "step": "retry_shell", "phase": "detail", "status": "info",
                    "message": f"ℹ️ 详情页为精简版，{wait:.0f}s 后重试全量页（{attempt + 1}/{max_retries + 1}）",
                    "current": 0, "total": 0,
                    "extra": {"attempt": attempt + 1},
                })
            logger.warning(f"[{label}] 拿到 SEO 降级壳页，{wait:.1f}s 后重试（{attempt+1}/{max_retries+1}）")
            await _smart_sleep(wait, CRAWLER["jitter"])
        else:
            logger.warning(f"[{label}] 多次重试仍为降级壳页，按抓取失败处理")

    # 重试耗尽仍拿不到全量页：按失败处理（不占目标名额，会继续翻页补位），
    # 而不是把长文误判成「正文过短」
    return None


async def _fetch_discuss(client: httpx.AsyncClient, content_id: str, progress_cb=None) -> Optional[dict]:
    headers = _build_headers(content_type=False)
    headers["Referer"] = f"https://www.nowcoder.com/discuss/{content_id}"
    url = f"{CRAWLER['discuss_api']}/{content_id}"
    resp, status, elapsed = await _get_with_retry(
        client, url, headers, progress_cb=progress_cb, label=f"discuss_{content_id[:8]}",
    )
    if resp is None:
        return None

    try:
        data = resp.json()
    except Exception:
        return None
    if not data.get("success"):
        return None

    cd = data.get("data", {})
    rich = cd.get("richText", "") or cd.get("content", "")
    content = html_to_text(rich)
    return {
        "title": cd.get("title", ""),
        "content": content,
        "url": f"{CRAWLER['discuss_url']}/{content_id}",
    }


async def fetch_detail(post_meta: dict, progress_cb=None, proxy: Optional[str] = None) -> Optional[dict]:
    rc_type = post_meta.get("rc_type")
    async with httpx.AsyncClient(
        timeout=CRAWLER["timeout"] + 5,
        follow_redirects=True,
        cookies=_active_cookies(),
        proxy=proxy,
    ) as client:
        if rc_type == 201:
            return await _fetch_feed(client, post_meta["uuid"], progress_cb)
        elif rc_type == 207:
            return await _fetch_discuss(client, post_meta["content_id"], progress_cb)
    return None


# ============ 主入口 ============

async def crawl_keyword(
    query: str,
    days_filter: Optional[int] = 30,
    max_pages: int = 5,
    max_posts: int = 30,
    delay: Optional[float] = None,
    progress_cb=None,
    stop_event: Optional[asyncio.Event] = None,
) -> list[dict]:
    """
    完整流程：搜索 → 抓取详情 → 返回结构化面经列表

    stop_event: 置位后协作式停止——尽快结束抓取并返回已收集到的面经，
                已抓到的有效面经不丢失（由调用方入库）。
    """
    invalidate_cookie_cache()
    base_delay = delay if delay is not None else CRAWLER["delay"]

    # 会话粘性：本次任务只选一次代理，搜索 + 所有详情共用同一个出口 IP。
    # 单 cookie 下频繁换 IP 是典型风控特征，轮换只发生在任务之间。
    proxy_url = pick_proxy()
    proxy_note = f"代理锁定 {_mask_proxy(proxy_url)}" if proxy_url else "直连"

    # 同一任务固定 UA，与代理 IP 组成一致的浏览器指纹
    _TASK_UA.set(_random_ua())

    if progress_cb:
        progress_cb({
            "step": "start", "phase": "search", "status": "success",
            "message": f"🚀 开始爬取: {query}（近{days_filter}天，最多{max_posts}篇）· {proxy_note}",
            "current": 0, "total": 0,
            "extra": {"query": query, "cookies_loaded": bool(_active_cookies()),
                      "proxy": _mask_proxy(proxy_url) if proxy_url else None},
        })

    # 边搜边抓：max_posts 是「最终有效面经」的目标数。
    # 正文过短 / 抓取失败都不占名额，继续翻下一页补充，直到拿够 target 或翻完 max_pages。
    target = max_posts
    min_len = CRAWLER["min_content_length"]
    collected = []
    failed_count = 0
    skipped_short = 0
    processed = 0
    stopped = False

    def emit(step: str, status: str, message: str, extra: dict | None = None):
        if progress_cb:
            progress_cb({
                "step": step, "phase": "detail", "status": status,
                "message": message,
                # 进度条以「已拿到的有效条数 / 目标」推进
                "current": min(len(collected), target), "total": target,
                "extra": extra or {},
            })

    try:
        async for entries in iter_search_entries(
            query=query,
            max_pages=max_pages,
            days_filter=days_filter,
            progress_cb=progress_cb,
            proxy=proxy_url,
        ):
            if stop_event is not None and stop_event.is_set():
                stopped = True
                break
            for post in entries:
                if stop_event is not None and stop_event.is_set():
                    stopped = True
                    break
                processed += 1
                title = post.get("title", "")[:50]
                detail = await fetch_detail(post, progress_cb=progress_cb, proxy=proxy_url)

                if detail and len(detail.get("content", "")) > min_len:
                    item = {
                        "title": detail["title"] or post.get("title", ""),
                        "content": detail["content"],
                        "url": detail["url"],
                        "author": post.get("author", ""),
                        "publish_time": post.get("publish_time"),
                        "school": post.get("school", ""),
                        "position": post.get("position", ""),
                        "tags": post.get("tags", []),
                        "query": post.get("query", query),
                        "source": "nowcoder",
                    }
                    item["questions"] = extract_questions(item["content"])
                    collected.append(item)
                    emit("fetch_detail", "success",
                         f"✅ [{len(collected)}/{target}] {title}",
                         {"title": title, "questions_count": len(item["questions"])})
                elif detail is None:
                    failed_count += 1
                    emit("fetch_detail", "failed",
                         f"❌ [已检查 {processed} 条] {title}",
                         {"title": title})
                else:
                    # 正文过短：直接丢弃，不占目标名额，后续翻页补够
                    skipped_short += 1
                    emit("skip_short", "skipped",
                         f"⏭️ 正文过短，跳过并继续补充 {title}",
                         {"title": title, "reason": "content_too_short"})

                await _smart_sleep(base_delay, CRAWLER["jitter"])

            if len(collected) >= target:
                break
    except asyncio.CancelledError:
        # task.cancel() 立即打断当前请求 / sleep；已收集的面经照常返回
        stopped = True
        logger.info(f"爬取被取消（停止），已收集 {len(collected)} 条")

    # 完成
    if progress_cb:
        if stopped:
            done_status = "stopped"
            done_msg = (
                f"⏹️ 已手动停止：保留有效面经 {len(collected)} 条"
                f"（跳过短内容 {skipped_short} 条，失败 {failed_count} 条）"
            )
        elif len(collected) >= target:
            done_status = "success"
            done_msg = f"🏁 完成：有效面经 {len(collected)} 条（跳过短内容 {skipped_short} 条，失败 {failed_count} 条）"
        else:
            done_status = "success"
            done_msg = (
                f"🏁 已翻完 {max_pages} 页，共得有效面经 {len(collected)} / 目标 {target} 条"
                f"（跳过短内容 {skipped_short} 条，失败 {failed_count} 条）"
            )
        progress_cb({
            "step": "done", "phase": "complete", "status": done_status,
            "message": done_msg,
            "current": min(len(collected), target), "total": target,
            "extra": {
                "success": len(collected), "target": target,
                "failed": failed_count, "skipped_short": skipped_short,
                "stopped": stopped,
            },
        })

    return collected


# ============ 问题提取 ============

def extract_questions(content: str) -> list[str]:
    questions = []
    lines = content.split("\n")

    for line in lines:
        line = line.strip()
        if not line or len(line) < 5:
            continue

        m = re.match(r"^(\d{1,2})[\.、\)：:\s]+(.+)", line)
        if m:
            q = m.group(2).strip()
            if not re.match(r"^(答|答[:：]|回答|Ans|A:)", q, re.IGNORECASE):
                q = re.split(r"\s*[答A答:]\s*[:：]?", q)[0].strip()
                if q.endswith("?") or q.endswith("？") or len(q) > 15:
                    questions.append(q)
            continue

        m = re.match(r"^[Qq][:：\.、\s]+(.+)", line)
        if m:
            q = m.group(1).strip()
            if not re.match(r"^(答|回答|Ans|A:)", q, re.IGNORECASE):
                questions.append(q)

    return questions
