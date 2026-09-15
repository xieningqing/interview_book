"""
AI 答题链路（数据清洗之后的第二道处理）
- 仅针对已清洗面经的结构化问题
- 每篇面经一次模型调用，为每题生成「贴切、简洁」的参考答案
- 面经当事人的原回答原样保留（questions.answer），AI 答案单独存 questions.ai_answer，形成参照
- 面试背景 / 自我介绍 / 反问等无标准答案的题自动跳过
"""
import asyncio
import json
import logging
import re
from typing import Optional

from app.ai import llm
from app.ai.cleaner import _tidy_text
from app.storage import database as db

logger = logging.getLogger(__name__)

# 并发答题数：并行调用大模型加快批量处理；过大易触发限流，2 为稳妥值
CONCURRENCY = 2

SYSTEM_PROMPT = """你是资深技术面试考官，负责为面经中的面试题写「参考答案」。

要求：
1. 贴切：直接针对问题作答，结合题目的岗位与技术方向；问什么答什么，不发散、不铺垫背景。
2. 简洁：先给一句话核心结论，再用 2-4 个要点简短展开；一般每题不超过 150 字，复杂题也不超过 250 字。
3. 准确：只写有把握的通用知识；不要编造面经当事人的个人经历；原回答仅作语境参考，即使原回答缺失或答偏了，你也要正常给出正确答案。
4. 代码题写清关键思路或最核心的几行代码即可，不要大段实现。
5. 以下题目的 a 必须返回空字符串：
   - 面试基本情况、背景介绍、时间线、流程梳理；
   - 自我介绍、个人情况、HR 闲聊、反问环节、主观感受类；
   - 纯个人经历题（如「介绍一下你的项目」「你实习做了什么」）——但若问的是项目中的技术方案/原理，则正常作答。

只输出一个 JSON 对象，不要输出任何解释或 markdown 代码块：
{"answers": [{"i": 1, "a": "参考答案或空字符串"}]}
i 为题目序号（从 1 开始），顺序与输入一致，每道题都必须出现。"""

_QA_BLOCK_RE = re.compile(
    r"Q\s*(\d+)\s*[：:]\s*(.*?)\s*A\s*\1\s*[：:]\s*(.*?)(?=\n\s*Q\s*\d+\s*[：:]|\Z)",
    flags=re.DOTALL,
)


def parse_qa_content(content: str) -> list[dict]:
    """从清洗后的结构化正文（Q1：… A1：…）解析问答，用于旧数据回填原回答"""
    out = []
    for m in _QA_BLOCK_RE.finditer(content or ""):
        q = _tidy_text(m.group(2))
        a = _tidy_text(m.group(3))
        if q:
            out.append({"q": q, "a": a})
    return out


async def answer_interview(interview: dict, keyword_name: str) -> dict:
    """
    为单篇已清洗面经生成 AI 答案并落库。
    Returns: {"questions": n, "answered": n, "skipped": n}
    """
    inv_id = interview["id"]
    qa = db.get_qa_pairs(inv_id)

    # 旧版清洗数据 questions.answer 为空：从结构化正文解析后回填
    if qa and all(not (x.get("answer") or "").strip() for x in qa):
        parsed = parse_qa_content(interview.get("content") or "")
        if parsed:
            db.backfill_question_answers(inv_id, parsed)
            qa = db.get_qa_pairs(inv_id)

    if not qa:
        raise RuntimeError("没有可答题的结构化问题，请先清洗")

    lines = []
    lines.append(f"岗位/方向：{interview.get('position') or keyword_name or '未注明'}")
    if interview.get("title"):
        lines.append(f"面经标题：{interview['title']}")
    for i, item in enumerate(qa, 1):
        raw_a = (item.get("answer") or "").strip()
        if not raw_a or raw_a == "（面经中未给出明确答案）":
            raw_a = "（原回答缺失）"
        lines.append(f"{i}. 问：{item['question']}\n   原回答：{raw_a[:500]}")
    user_prompt = "\n".join(lines)

    data = await llm.chat_json(SYSTEM_PROMPT, user_prompt)

    reply = data.get("answers")
    if not isinstance(reply, list):
        raise RuntimeError("模型未返回 answers 列表")

    # 返回条目数不足题量三分之二视为输出截断/异常：不保存、不标记完成，交给批量重试
    min_entries = max(1, (len(qa) * 2 + 2) // 3)
    valid_entries = [x for x in reply if isinstance(x, dict)]
    if len(valid_entries) < min_entries:
        raise RuntimeError(
            f"模型只返回 {len(valid_entries)}/{len(qa)} 题答案，疑似输出截断，本次不保存"
        )

    # 按序号对齐到题目；缺失/越界序号退化为按顺序对齐
    by_index: dict[int, str] = {}
    for item in valid_entries:
        if isinstance(item.get("i"), int):
            by_index[item["i"]] = str(item.get("a") or "")

    save_items = []
    answered = 0
    skipped = 0
    for pos, q in enumerate(qa, 1):
        text = by_index.get(pos)
        if text is None and pos <= len(valid_entries):
            text = str(valid_entries[pos - 1].get("a") or "")
        text = (text or "").strip()
        if text:
            answered += 1
        else:
            skipped += 1
        save_items.append({"question_id": q["id"], "ai_answer": text})

    saved = db.save_ai_answers(inv_id, save_items)
    if not saved:
        # 答题等待期间该篇被重新清洗（questions 已重建）：不落时间戳，按失败处理
        raise RuntimeError("题目已被重新清洗，请重新答题")
    return {"questions": len(qa), "answered": answered, "skipped": skipped}


async def answer_batch(
    keyword_id: int,
    keyword_name: str,
    progress_cb=None,
    stop_event: Optional[asyncio.Event] = None,
) -> dict:
    """按关键词批量 AI 答题（增量，只处理已清洗未答题的面经）"""
    targets = db.list_answer_targets(keyword_id)
    stats = {"total": len(targets), "answered": 0, "questions": 0, "skipped": 0, "failed": 0}

    def emit(step, status, message, current, extra=None):
        if progress_cb:
            progress_cb({
                "step": step, "phase": "answer", "status": status,
                "message": message, "current": current, "total": len(targets),
                "extra": extra or {},
            })

    if not targets:
        emit("done", "success", "该关键词下没有待答题的已清洗面经", 0)
        return stats

    # 立即回一条开始事件：首篇模型调用耗时较长，避免前端一直停在「准备中」
    emit("start", "info",
         f"🚀 共 {len(targets)} 篇待答题，并发 {CONCURRENCY} 篇调用大模型（失败自动重试）", 0)

    # 并发执行：信号量限流，各篇完成后计数/发进度；任一篇失败只记该篇失败
    sem = asyncio.Semaphore(CONCURRENCY)
    processed = 0

    async def work(inv):
        nonlocal processed
        async with sem:
            title = (inv.get("title") or "")[:40]
            # 模型返回前先告知当前处理项，单篇耗时长时界面不会像卡住
            emit("answer_item", "info", f"⏳ 正在答题 {title}", processed)
            try:
                result = await answer_interview(inv, keyword_name)
            except Exception as e:
                # asyncio.CancelledError 是 BaseException，不会被这里捕获，取消正常传播
                stats["failed"] += 1
                logger.warning(f"AI 答题失败 id={inv['id']}: {e}")
                processed += 1
                emit("answer_item", "failed",
                     f"❌ {title} —— {str(e)[:80]}", processed,
                     {"title": title, "error": str(e)[:120], **stats})
            else:
                stats["answered"] += 1
                stats["questions"] += result["answered"]
                stats["skipped"] += result["skipped"]
                processed += 1
                emit("answer_item", "success",
                     f"✅ 已答题（{result['answered']} 题，跳过 {result['skipped']}）{title}", processed,
                     {"title": title, **stats})
            # 请求间留出间隔，降低限流概率（任务被取消时 sleep 抛 CancelledError 向上传播）
            await asyncio.sleep(0.3)

    pending = {asyncio.create_task(work(inv)) for inv in targets}
    try:
        while pending:
            if stop_event is not None and stop_event.is_set():
                for t in pending:
                    t.cancel()
                await asyncio.gather(*pending, return_exceptions=True)
                emit("done", "stopped",
                     f"⏹️ 已停止：答题 {stats['answered']} 篇，生成 {stats['questions']} 题，失败 {stats['failed']}",
                     processed, {"stopped": True, **stats})
                return stats
            _, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
    except asyncio.CancelledError:
        # main.py 停止/重启会 cancel 本任务：取消未完成的子任务后继续抛出
        for t in pending:
            t.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        raise

    emit("done", "success",
         f"🏁 答题完成：{stats['answered']} 篇，生成答案 {stats['questions']} 题，跳过 {stats['skipped']}，失败 {stats['failed']}",
         processed, stats)
    return stats
