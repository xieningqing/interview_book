"""
面经数据清洗
- 调用大模型判断面经与搜索关键词是否相关（搜 agent 却是 Java 岗 → 标记不相关）
- 相关面经：把松散正文整理成结构化 Q&A，答案完整保留，问题去重归并
- 清洗结果落库：原文备份 raw_content，正文重写，questions 重建
"""
import asyncio
import json
import logging
import re
from typing import Optional

from app.ai import llm
from app.storage import database as db

logger = logging.getLogger(__name__)

# 送给模型的正文上限（字符），兼顾长文与 token 成本
MAX_CONTENT_CHARS = 12000

SYSTEM_PROMPT = """你是牛客面经数据清洗助手。你要处理的最小单位是「整篇面经」，不是单个问题。

1. 相关性判定（整篇粒度，只有 true / false 两种结果）：
   - 通读全文后判断「整篇面经」与给定搜索关键词是否有实质关联。
   - 只有当全文几乎都是其它方向（如关键词是「agent/智能体/大模型应用」，整篇却是普通 Java CRUD 校招）时，才判 relevant=false。
   - 只要主体内容与关键词相关（哪怕夹带了少数 Java/八股等其它问题），整篇都算 relevant=true。
   - 严禁因为其中某一个问题看似不相关就把整篇判为不相关，也不要删除相关面经里的个别问题。
2. 对 relevant=true 的面经，只做「轻量排版」，把杂乱正文整理成清晰的结构化问答。
   【核心原则：排版优先，内容最小改动】
   - 只理顺格式、分段与明显口误，不要润色、不要改写措辞、不要替作者总结或升华为书面语，尽量保留原文的原话、语气和表达习惯；
   - 保留原文的趣味性内容：吐槽、心情、面试氛围、面试官反应、小故事、梗等一律保留，可单独成段或并入相关问答，不要当废话删掉；
   - 保留原文全部信息点，严禁增删观点、严禁编造，也不要删减八股/夹带的其它方向问题。
   【整理动作（仅这些）】
   - 识别问答边界，按「背景 → 项目/实习 → 技术问题 → 反问/HR」大致归拢，语义明显重复的相邻问答可合并；
   - 开头的时间、公司、岗位、轮次等背景归为第一条「面试基本情况」；
   - 问题补成通顺问句（可保留口语），去掉 Q:/问：等前缀；一题一问。
   【答案排版】
   - 用原文原句组织，仅做断句、分行、标点修正；多个要点自然分行（可加 1. 2. 或 - ），但不为了编号而改写内容；
   - 明显的错别字可修正，「然后、就是」等只在严重影响阅读时少量清理，不必追求干净而丢失口语感；
   - 代码/命令/配置原样保留。
   原文没有给出明确答案的问题，answer 填「（面经中未给出明确答案）」。
   只有与面试完全无关的纯广告、刷屏内容才丢弃；自我介绍若带个人背景/趣味信息也保留。

只输出一个 JSON 对象，不要输出任何解释或 markdown 代码块，格式：
{"relevant": true 或 false, "qa": [{"q": "问题", "a": "答案"}]}
不相关时 qa 输出空数组。"""


def _extract_json(text: str) -> Optional[dict]:
    """从模型输出中容错提取 JSON 对象"""
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


def _tidy_text(text: str) -> str:
    """规整模型输出的空白与换行：逐行去首尾空白、压缩 3+ 换行为 1 个空行"""
    lines = []
    for line in (text or "").splitlines():
        line = line.strip()
        if line:
            lines.append(line)
    return "\n".join(lines)


def _format_content(qa: list[dict]) -> str:
    """把 Q&A 列表渲染为结构化正文文本，并做轻量排版规整"""
    parts = []
    for i, item in enumerate(qa, 1):
        q = _tidy_text(item.get("q") or "")
        a = _tidy_text(item.get("a") or "") or "（面经中未给出明确答案）"
        parts.append(f"Q{i}：{q}\nA{i}：{a}")
    return "\n\n".join(parts)


def _normalize_qa(data: dict) -> list[dict]:
    qa = data.get("qa") or []
    out = []
    seen = set()
    for item in qa:
        if not isinstance(item, dict):
            continue
        q = str(item.get("q", "")).strip()
        a = str(item.get("a", "")).strip()
        if not q:
            continue
        key = re.sub(r"\s+", "", q).lower()
        if key in seen:
            continue
        seen.add(key)
        out.append({"q": q, "a": a})
    return out


async def clean_one(interview: dict, keyword: str) -> dict:
    """
    清洗单篇面经并落库。

    Returns: {"action": "cleaned"|"irrelevant", "questions": n}
    """
    content = interview.get("content") or ""
    user_prompt = (
        f"搜索关键词：{keyword}\n"
        f"面经标题：{interview.get('title','')}\n"
        f"面经正文：\n{content[:MAX_CONTENT_CHARS]}"
    )

    raw = await llm.chat(SYSTEM_PROMPT, user_prompt)
    data = _extract_json(raw)
    if data is None:
        raise RuntimeError("模型返回不是合法 JSON")

    if not data.get("relevant", False):
        db.mark_interview_irrelevant(interview["id"], True)
        return {"action": "irrelevant", "questions": 0}

    qa = _normalize_qa(data)
    if not qa:
        # 模型判了相关却没给出任何问答：保守起见标记失败，不动原文
        raise RuntimeError("模型判定相关但未返回问答内容")

    cleaned = _format_content(qa)
    db.save_cleaned_interview(interview["id"], cleaned, qa)
    return {"action": "cleaned", "questions": len(qa)}


async def clean_batch(
    keyword_id: int,
    keyword_name: str,
    progress_cb=None,
    stop_event: Optional[asyncio.Event] = None,
) -> dict:
    """按关键词批量清洗，返回 {total, cleaned, irrelevant, failed}"""
    targets = db.list_clean_targets(keyword_id)
    stats = {"total": len(targets), "cleaned": 0, "irrelevant": 0, "failed": 0}

    def emit(step, status, message, current, extra=None):
        if progress_cb:
            progress_cb({
                "step": step, "phase": "clean", "status": status,
                "message": message, "current": current, "total": len(targets),
                "extra": extra or {},
            })

    if not targets:
        emit("done", "success", "该关键词下没有需要清洗的面经", 0)
        return stats

    for i, inv in enumerate(targets, 1):
        if stop_event is not None and stop_event.is_set():
            emit("done", "stopped",
                 f"⏹️ 已停止：清洗 {stats['cleaned']}，不相关 {stats['irrelevant']}，失败 {stats['failed']}",
                 i - 1, {"stopped": True, **stats})
            return stats

        title = (inv.get("title") or "")[:40]
        try:
            result = await clean_one(inv, keyword_name)
            if result["action"] == "irrelevant":
                stats["irrelevant"] += 1
                emit("clean_item", "skipped",
                     f"🏷️ 与「{keyword_name}」不相关，已标记 {title}", i,
                     {"title": title, "action": "irrelevant"})
            else:
                stats["cleaned"] += 1
                emit("clean_item", "success",
                     f"✅ [{i}/{len(targets)}] 已清洗（{result['questions']} 问）{title}", i,
                     {"title": title, "questions": result["questions"]})
        except Exception as e:
            stats["failed"] += 1
            logger.warning(f"清洗失败 id={inv['id']}: {e}")
            emit("clean_item", "failed",
                 f"❌ [{i}/{len(targets)}] {title} —— {str(e)[:80]}", i,
                 {"title": title, "error": str(e)[:120]})

        # 请求间留出间隔，降低限流概率
        await asyncio.sleep(0.5)

    emit("done", "success",
         f"🏁 清洗完成：有效 {stats['cleaned']}，不相关 {stats['irrelevant']}，失败 {stats['failed']}",
         len(targets), stats)
    return stats
