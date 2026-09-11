"""
FastAPI 主入口
- Jinja2 模板页面路由
- REST API 路由
- 异步爬虫任务调度
"""
import asyncio
import logging
import uuid
from datetime import datetime
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import Optional

from app.config import BASE_DIR, CRAWLER
from app.storage import database as db
from app.storage import file_store
from app.storage import cookie_manager
from app.storage import proxy_manager
from app.crawler import nowcoder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ============ 应用生命周期 ============

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时初始化数据库
    db.init_db()
    yield
    # 关闭时清理
    pass


app = FastAPI(title="面经爬虫管理系统", lifespan=lifespan)

# 静态资源
app.mount(
    "/static",
    StaticFiles(directory=str(BASE_DIR / "app" / "static")),
    name="static",
)

# 模板
templates = Jinja2Templates(directory=str(BASE_DIR / "app" / "templates"))


# ============ Pydantic 模型 ============

class KeywordCreate(BaseModel):
    name: str
    days_filter: int = 30
    max_pages: int = 5
    max_posts: int = 30


class CrawlRequest(BaseModel):
    keyword_id: Optional[int] = None
    query: Optional[str] = None
    days_filter: Optional[int] = None
    max_pages: Optional[int] = None
    max_posts: Optional[int] = None


class CookieUpdate(BaseModel):
    """更新 Cookie 支持两种格式: string (header 格式) 或 dict (key-value)"""
    cookie_string: Optional[str] = None
    cookies: Optional[dict] = None


class ProxyUpdate(BaseModel):
    """更新代理配置"""
    proxies: Optional[list[str]] = None
    proxy_text: Optional[str] = None        # 每行一个代理地址
    enabled: Optional[bool] = None
    strategy: Optional[str] = None          # round_robin / random / first


# ============ 页面路由 ============

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """首页：关键词管理 + 统计"""
    stats = db.get_stats()
    keywords = db.list_keywords()
    return templates.TemplateResponse(
        request, "index.html",
        {"stats": stats, "keywords": keywords},
    )


@app.get("/interviews", response_class=HTMLResponse)
async def interviews_page(request: Request, keyword_id: Optional[int] = None):
    """面经列表页"""
    keywords = db.list_keywords()
    interviews = db.list_interviews(keyword_id=keyword_id, limit=500)
    selected_kw = db.get_keyword(keyword_id) if keyword_id else None
    return templates.TemplateResponse(
        request, "interviews.html",
        {
            "interviews": interviews,
            "keywords": keywords,
            "selected_keyword": selected_kw,
        },
    )


@app.get("/interviews/{interview_id}", response_class=HTMLResponse)
async def interview_detail(request: Request, interview_id: int):
    """面经详情页"""
    inv = db.get_interview(interview_id)
    if not inv:
        raise HTTPException(status_code=404, detail="面经不存在")
    return templates.TemplateResponse(
        request, "detail.html",
        {"interview": inv},
    )


# ============ API 路由 ============

@app.get("/api/stats")
async def api_stats():
    return db.get_stats()


@app.get("/api/keywords")
async def api_list_keywords():
    return db.list_keywords()


@app.post("/api/keywords")
async def api_add_keyword(data: KeywordCreate):
    kid = db.add_keyword(
        name=data.name.strip(),
        days_filter=data.days_filter,
        max_pages=data.max_pages,
        max_posts=data.max_posts,
    )
    return {"id": kid, "message": "添加成功"}


@app.delete("/api/keywords/{keyword_id}")
async def api_delete_keyword(keyword_id: int):
    db.delete_keyword(keyword_id)
    return {"message": "删除成功"}


@app.get("/api/interviews")
async def api_list_interviews(keyword_id: Optional[int] = None, limit: int = 100):
    return db.list_interviews(keyword_id=keyword_id, limit=limit)


@app.get("/api/interviews/{interview_id}")
async def api_get_interview(interview_id: int):
    inv = db.get_interview(interview_id)
    if not inv:
        raise HTTPException(status_code=404, detail="面经不存在")
    return inv


@app.delete("/api/interviews/{interview_id}")
async def api_delete_interview(interview_id: int):
    db.delete_interview(interview_id)
    return {"message": "删除成功"}


# ============ 爬取任务 ============

# 存储任务进度（结构化 steps，供前端可视化）
# 每个 step: {phase, status, message, current, total, extra, timestamp}
task_status = {}
# 运行中的后台任务：task_id -> asyncio.Task，用于手动停止
running_tasks: dict[str, asyncio.Task] = {}
# 每个任务的协作式停止信号
stop_events: dict[str, asyncio.Event] = {}


@app.post("/api/crawl")
async def api_crawl(req: CrawlRequest):
    """
    触发爬取任务（异步执行，通过 /api/task/{task_id}/progress 查询进度）

    支持两种方式:
      1. keyword_id: 已保存的关键词配置
      2. query + 可选参数: 临时关键词直接爬取
    """
    task_id = f"task_{uuid.uuid4().hex[:16]}"
    started_at = datetime.now().isoformat()
    progress = {
        "task_id": task_id,
        "status": "running",
        "steps": [],
        "stats": {
            "search_pages": 0,
            "detail_total": 0,   # 参数解析后再设置为最终 max_posts
            "detail_success": 0,
            "detail_failed": 0,
            "detail_skipped": 0,
            "risk_hits": 0,
        },
        "result": None,
    }
    task_status[task_id] = progress

    # 解析参数
    keyword_obj = None
    query = req.query
    days_filter = req.days_filter
    max_pages = req.max_pages
    max_posts = req.max_posts
    keyword_id = req.keyword_id

    if req.keyword_id:
        keyword_obj = db.get_keyword(req.keyword_id)
        if not keyword_obj:
            raise HTTPException(status_code=404, detail="关键词不存在")
        query = keyword_obj["name"]
        # 注意用 is None 判断：days_filter=0 表示「不限时间」，不能被 or 吞掉
        if days_filter is None:
            days_filter = keyword_obj["days_filter"]
        if max_pages is None:
            max_pages = keyword_obj["max_pages"]
        if max_posts is None:
            max_posts = keyword_obj["max_posts"]
    elif not query:
        raise HTTPException(status_code=400, detail="需要提供 keyword_id 或 query")

    if days_filter is None:
        days_filter = 30
    if max_pages is None:
        max_pages = 5
    if max_posts is None:
        max_posts = 30
    progress["stats"]["detail_total"] = max_posts

    # 后台执行爬取
    async def run_crawl():
        def on_progress(step: dict):
            """爬虫回调：接收结构化 step dict"""
            progress["steps"].append(step)
            # 实时更新统计
            s = step.get("status")
            ph = step.get("phase")
            if ph == "search" and step.get("step") == "search_page":
                progress["stats"]["search_pages"] += 1
            elif ph == "detail" and step.get("step") == "fetch_detail":
                if s == "success": progress["stats"]["detail_success"] += 1
                elif s == "failed": progress["stats"]["detail_failed"] += 1
                # 正文过短（skip_short）只出现在请求日志里，不累加任何条数；
                # 爬虫会继续翻页补充直到拿满 detail_total 个有效条
            if s == "risk":
                progress["stats"]["risk_hits"] += 1

        stop_event = stop_events[task_id]
        try:
            # 爬取
            interviews = await nowcoder.crawl_keyword(
                query=query,
                days_filter=days_filter,
                max_pages=max_pages,
                max_posts=max_posts,
                delay=CRAWLER["delay"],
                progress_cb=on_progress,
                stop_event=stop_event,
            )

            # 最后一条 done 步骤带 stopped 标记（用户手动停止）
            last_done = next(
                (s for s in reversed(progress["steps"]) if s.get("step") == "done"), None
            )
            manually_stopped = bool(last_done and last_done.get("extra", {}).get("stopped"))

            # 保存到数据库（手动停止时已抓到的面经也照常保留）
            kid_to_save = keyword_id
            if not kid_to_save:
                kid_to_save = db.add_keyword(query, days_filter, max_pages, max_posts)

            new_count, existing, dup_count = db.save_interviews(kid_to_save, interviews)
            db.update_keyword_crawled_at(kid_to_save)

            # 保存到 TXT
            txt_paths = file_store.save_batch_to_txt(interviews, query)

            progress["status"] = "stopped" if manually_stopped else "done"
            progress["result"] = {
                "total_found": len(interviews),
                "new_saved": new_count,
                "already_exists": len(existing),
                "dup_titles": dup_count,
                "txt_saved": len(txt_paths),
            }

        except Exception as e:
            logger.exception("爬取失败")
            progress["status"] = "failed"
            progress["steps"].append({
                "step": "error", "phase": "complete", "status": "failed",
                "message": f"❌ 异常: {str(e)}",
                "current": 0, "total": 0,
            })
        finally:
            # 任务结束后持久化到数据库
            try:
                db.save_crawl_task(
                    task_id=task_id,
                    query=query,
                    keyword_id=keyword_id,
                    days_filter=days_filter,
                    max_pages=max_pages,
                    max_posts=max_posts,
                    status=progress["status"],
                    stats=progress["stats"],
                    result=progress.get("result"),
                    steps=progress["steps"],
                    started_at=started_at,
                    finished_at=datetime.now().isoformat(),
                )
            except Exception as e:
                logger.warning(f"持久化任务失败: {e}")
            running_tasks.pop(task_id, None)
            stop_events.pop(task_id, None)

    # 启动后台任务并保存引用，供手动停止使用
    stop_events[task_id] = asyncio.Event()
    running_tasks[task_id] = asyncio.create_task(run_crawl())

    return {"task_id": task_id, "status": "running", "message": "爬取任务已启动"}


@app.post("/api/task/{task_id}/stop")
async def api_stop_task(task_id: str):
    """手动停止运行中的爬取任务；已抓取的面经会保留入库"""
    task = running_tasks.get(task_id)
    if not task or task.done():
        raise HTTPException(status_code=404, detail="任务不存在或已结束")
    # 置位协作信号 + cancel 立即打断当前请求/睡眠；爬虫会在收尾后正常返回
    ev = stop_events.get(task_id)
    if ev is not None:
        ev.set()
    task.cancel()
    return {"message": "正在停止…已抓取的面经会保留"}


@app.get("/api/task/{task_id}/progress")
async def api_task_progress(task_id: str):
    """查询任务进度"""
    progress = task_status.get(task_id)
    if not progress:
        raise HTTPException(status_code=404, detail="任务不存在")
    return progress


@app.post("/api/crawl-all")
async def api_crawl_all():
    """触发所有已保存关键词的爬取任务"""
    keywords = db.list_keywords()
    results = []
    for kw in keywords:
        req = CrawlRequest(keyword_id=kw["id"])
        result = await api_crawl(req)
        results.append({"keyword": kw["name"], "task_id": result["task_id"]})
    return {"tasks": results}


# ============ Cookie 管理 ============

@app.get("/cookie", response_class=HTMLResponse)
async def cookie_page(request: Request):
    """Cookie 配置页面"""
    status = cookie_manager.get_cookie_status()
    return templates.TemplateResponse(
        request, "cookie.html",
        {"cookie_status": status},
    )


@app.get("/api/cookie")
async def api_get_cookie():
    """查询当前 Cookie 配置状态（脱敏后返回）"""
    return cookie_manager.get_cookie_status()


@app.post("/api/cookie")
async def api_update_cookie(data: CookieUpdate):
    """
    更新 Cookie

    支持两种传参方式:
      1. { "cookie_string": "a=1; b=2; token=xxx" }   —— 直接贴浏览器的 Cookie header
      2. { "cookies": { "a": "1", "b": "2" } }       —— 结构化 key-value
    """
    if data.cookie_string:
        normalized = cookie_manager.save_cookies(data.cookie_string.strip())
    elif data.cookies:
        normalized = cookie_manager.save_cookies(data.cookies)
    else:
        raise HTTPException(status_code=400, detail="需要提供 cookie_string 或 cookies")

    # 刷新爬虫的 Cookie 缓存
    nowcoder.invalidate_cookie_cache()

    status = cookie_manager.get_cookie_status()
    return {
        "message": f"✓ Cookie 已保存（{len(normalized)} 个）",
        "count": len(normalized),
        "configured": status["configured"],
    }


@app.delete("/api/cookie")
async def api_clear_cookie():
    """清除 Cookie"""
    cookie_manager.clear_cookies()
    nowcoder.invalidate_cookie_cache()
    return {"message": "✓ Cookie 已清除"}


# ============ 代理管理 ============

@app.get("/proxy", response_class=HTMLResponse)
async def proxy_page(request: Request):
    """代理配置页面"""
    status = proxy_manager.get_proxy_status()
    return templates.TemplateResponse(
        request, "proxy.html",
        {"proxy_status": status},
    )


@app.get("/api/proxy")
async def api_get_proxy():
    return proxy_manager.get_proxy_status()


@app.post("/api/proxy")
async def api_update_proxy(data: ProxyUpdate):
    """更新代理配置"""
    # 收集代理地址列表
    proxy_list = []
    if data.proxies:
        proxy_list.extend(data.proxies)
    if data.proxy_text:
        proxy_list.extend(
            [line.strip() for line in data.proxy_text.splitlines() if line.strip()]
        )

    # 去重
    proxy_list = list(dict.fromkeys(proxy_list))

    if not proxy_list and data.enabled is not False:
        raise HTTPException(status_code=400, detail="至少提供一个代理地址")

    if proxy_list:
        proxy_manager.set_proxies(
            proxy_list,
            enabled=data.enabled if data.enabled is not None else True,
            strategy=data.strategy or "round_robin",
        )

    if data.strategy and proxy_list:
        # strategy 已在 set_proxies 里处理
        pass
    elif data.enabled is not None and not proxy_list:
        proxy_manager.toggle_enabled(data.enabled)

    return {"message": "✓ 代理配置已更新"}


@app.post("/api/proxy/test")
async def api_test_proxy():
    """快速测试第一个代理是否可用"""
    cfg = proxy_manager.load_proxies()
    if not cfg.get("proxies"):
        return {"ok": False, "message": "未配置代理"}

    import httpx
    from app.storage.proxy_manager import _mask_proxy
    proxy_url = cfg["proxies"][0]
    masked = _mask_proxy(proxy_url)
    try:
        async with httpx.AsyncClient(proxy=proxy_url, timeout=10) as client:
            resp = await client.get("https://httpbin.org/ip", follow_redirects=True)
            data = resp.json()
            return {"ok": True, "message": f"代理可用，出口 IP: {data.get('origin', '?')}", "proxy": masked}
    except Exception as e:
        return {"ok": False, "message": f"代理失败: {str(e)[:100]}", "proxy": masked}


@app.delete("/api/proxy")
async def api_clear_proxy():
    proxy_manager.clear_proxies()
    return {"message": "✓ 代理已清除"}


# ============ 爬取任务历史 ============

@app.get("/tasks", response_class=HTMLResponse)
async def tasks_page(request: Request):
    """历史任务列表页面"""
    tasks = db.list_crawl_tasks(limit=100)
    return templates.TemplateResponse(
        request, "tasks.html",
        {"tasks": tasks},
    )


@app.get("/api/tasks")
async def api_list_tasks(limit: int = 50):
    return db.list_crawl_tasks(limit=limit)


@app.get("/api/tasks/{task_id}")
async def api_get_task(task_id: str):
    task = db.get_crawl_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")
    return task


@app.delete("/api/tasks/{task_id}")
async def api_delete_task(task_id: str):
    db.delete_crawl_task(task_id)
    return {"message": "已删除"}


@app.delete("/api/tasks")
async def api_clear_tasks():
    db.clear_all_crawl_tasks()
    return {"message": "已清空全部任务日志"}
