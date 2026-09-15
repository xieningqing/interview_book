"""
SQLite 数据库操作
存储结构化的关键词和面经数据
"""
import sqlite3
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

from app.config import DB_PATH

logger = logging.getLogger(__name__)


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db():
    """初始化数据库表"""
    conn = get_connection()
    try:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS keywords (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                days_filter INTEGER DEFAULT 30,
                max_pages INTEGER DEFAULT 5,
                max_posts INTEGER DEFAULT 30,
                created_at TEXT NOT NULL,
                last_crawled_at TEXT,
                enabled INTEGER DEFAULT 1
            );

            CREATE TABLE IF NOT EXISTS interviews (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                url TEXT UNIQUE NOT NULL,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                author TEXT,
                publish_time TEXT,
                school TEXT,
                position TEXT,
                tags TEXT,
                keyword_id INTEGER,
                source TEXT DEFAULT 'nowcoder',
                created_at TEXT NOT NULL,
                raw_content TEXT,
                cleaned_at TEXT,
                answered_at TEXT,
                is_irrelevant INTEGER DEFAULT 0,
                FOREIGN KEY (keyword_id) REFERENCES keywords(id)
            );

            CREATE TABLE IF NOT EXISTS questions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                interview_id INTEGER NOT NULL,
                question TEXT NOT NULL,
                answer TEXT,
                ai_answer TEXT,
                FOREIGN KEY (interview_id) REFERENCES interviews(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_interviews_publish_time ON interviews(publish_time);
            CREATE INDEX IF NOT EXISTS idx_interviews_keyword ON interviews(keyword_id);
            CREATE INDEX IF NOT EXISTS idx_questions_interview ON questions(interview_id);

            CREATE TABLE IF NOT EXISTS crawl_tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id TEXT UNIQUE NOT NULL,
                query TEXT NOT NULL,
                keyword_id INTEGER,
                days_filter INTEGER,
                max_pages INTEGER,
                max_posts INTEGER,
                status TEXT NOT NULL,
                stats TEXT,
                result TEXT,
                steps TEXT,
                started_at TEXT NOT NULL,
                finished_at TEXT,
                FOREIGN KEY (keyword_id) REFERENCES keywords(id) ON DELETE SET NULL
            );

            CREATE TABLE IF NOT EXISTS ai_tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id TEXT UNIQUE NOT NULL,
                kind TEXT NOT NULL,
                keyword_id INTEGER,
                keyword_name TEXT NOT NULL,
                status TEXT NOT NULL,
                stats TEXT,
                steps TEXT,
                started_at TEXT NOT NULL,
                finished_at TEXT,
                FOREIGN KEY (keyword_id) REFERENCES keywords(id) ON DELETE SET NULL
            );

            CREATE INDEX IF NOT EXISTS idx_ai_tasks_started ON ai_tasks(started_at);
        """)
        # 旧库平滑升级：补齐后续版本新增的列
        existed = {r[1] for r in conn.execute("PRAGMA table_info(interviews)")}
        for col, decl in (
            ("raw_content", "TEXT"),
            ("cleaned_at", "TEXT"),
            ("answered_at", "TEXT"),
            ("is_irrelevant", "INTEGER DEFAULT 0"),
        ):
            if col not in existed:
                conn.execute(f"ALTER TABLE interviews ADD COLUMN {col} {decl}")
        q_existed = {r[1] for r in conn.execute("PRAGMA table_info(questions)")}
        for col, decl in (
            ("answer", "TEXT"),
            ("ai_answer", "TEXT"),
        ):
            if col not in q_existed:
                conn.execute(f"ALTER TABLE questions ADD COLUMN {col} {decl}")
        conn.commit()
        logger.info("✓ 数据库初始化完成")
    finally:
        conn.close()


# ========== 关键词 CRUD ==========

def add_keyword(name: str, days_filter: int = 30, max_pages: int = 5, max_posts: int = 30) -> int:
    """保存关键词：同名则更新其参数（避免重名被静默丢弃）"""
    conn = get_connection()
    try:
        row = conn.execute("SELECT id FROM keywords WHERE name = ?", (name,)).fetchone()
        if row:
            conn.execute(
                "UPDATE keywords SET days_filter=?, max_pages=?, max_posts=? WHERE id=?",
                (days_filter, max_pages, max_posts, row["id"]),
            )
            conn.commit()
            return row["id"]
        cur = conn.execute(
            "INSERT INTO keywords (name, days_filter, max_pages, max_posts, created_at) VALUES (?, ?, ?, ?, ?)",
            (name, days_filter, max_pages, max_posts, datetime.now().isoformat()),
        )
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def get_keyword(id: int) -> Optional[dict]:
    conn = get_connection()
    try:
        row = conn.execute("SELECT * FROM keywords WHERE id = ?", (id,)).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def list_keywords() -> list[dict]:
    conn = get_connection()
    try:
        rows = conn.execute("SELECT * FROM keywords ORDER BY created_at DESC").fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def update_keyword_crawled_at(id: int):
    conn = get_connection()
    try:
        conn.execute(
            "UPDATE keywords SET last_crawled_at = ? WHERE id = ?",
            (datetime.now().isoformat(), id)
        )
        conn.commit()
    finally:
        conn.close()


def delete_keyword(id: int):
    conn = get_connection()
    try:
        # 先删除关联的面经和问题
        interviews = conn.execute("SELECT id FROM interviews WHERE keyword_id = ?", (id,)).fetchall()
        for inv in interviews:
            conn.execute("DELETE FROM questions WHERE interview_id = ?", (inv["id"],))
        conn.execute("DELETE FROM interviews WHERE keyword_id = ?", (id,))
        # 历史爬取日志保留，仅解除关联（兼容未带 ON DELETE SET NULL 的旧库表结构）
        conn.execute("UPDATE crawl_tasks SET keyword_id = NULL WHERE keyword_id = ?", (id,))
        conn.execute("DELETE FROM keywords WHERE id = ?", (id,))
        conn.commit()
    finally:
        conn.close()


# ========== 面经存储 ==========

# 标题归一化：去掉空白与常见标点后再比对，避免"换标点重发"绕过判重
_TITLE_NOISE_RE = re.compile(
    r"[\s\u3000【】\[\]（）()《》<>「」『』！？!?。，,、：:；;·\-—_|~～]+"
)


def normalize_title(title: str) -> str:
    """归一化标题用于判重：去标点/空白、转小写"""
    return _TITLE_NOISE_RE.sub("", title or "").lower()


def save_interviews(keyword_id: int, interviews: list[dict]) -> tuple[int, list[dict], int]:
    """
    批量保存面经：
    - URL 已存在 → 视为重复爬取，跳过
    - 同批次内重复标题（同一篇面经被收录多个 URL）→ 跳过，不重复入库
    - 库中同关键词已有同标题 → 入库并打「重复标题」tag
    返回 (新增数量, 已存在列表, 重复标题数量)
    """
    conn = get_connection()
    new_count = 0
    dup_count = 0
    existing = []

    try:
        # 库中该关键词下已有的归一化标题集合
        known_titles = set()
        for r in conn.execute(
            "SELECT title FROM interviews WHERE keyword_id = ?", (keyword_id,)
        ).fetchall():
            nt = normalize_title(r["title"])
            if nt:
                known_titles.add(nt)
        # 本批次内已接收的标题：同一批里重复出现视为同一篇，直接跳过
        batch_titles = set()

        for inv in interviews:
            url = inv.get("url", "")
            content = inv.get("content", "")
            title = inv.get("title", "")

            # 检查是否已存在（按 URL）
            existing_row = conn.execute("SELECT id FROM interviews WHERE url = ?", (url,)).fetchone()
            if existing_row:
                existing.append({"url": url, "title": title})
                continue

            nt = normalize_title(title)
            if nt and nt in batch_titles:
                # 同一批次内重复标题：不再入库，避免清洗/答题时同一篇处理两遍
                dup_count += 1
                existing.append({"url": url, "title": title})
                continue
            if nt:
                batch_titles.add(nt)

            # 与库内已有同标题相比，属跨批次重复：入库但打 tag
            tags = list(inv.get("tags", []) or [])
            if nt and nt in known_titles:
                if "重复标题" not in tags:
                    tags.append("重复标题")
                dup_count += 1

            # 插入面经
            conn.execute(
                """INSERT INTO interviews
                   (url, title, content, author, publish_time, school, position, tags, keyword_id, source, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    url, title, content,
                    inv.get("author", ""),
                    inv.get("publish_time").isoformat() if inv.get("publish_time") else None,
                    inv.get("school", ""),
                    inv.get("position", ""),
                    json.dumps(tags, ensure_ascii=False),
                    keyword_id,
                    inv.get("source", "nowcoder"),
                    datetime.now().isoformat(),
                )
            )
            interview_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

            # 插入问题列表
            questions = inv.get("questions", [])
            if questions:
                conn.executemany(
                    "INSERT INTO questions (interview_id, question) VALUES (?, ?)",
                    [(interview_id, q) for q in questions]
                )

            new_count += 1

        conn.commit()
        logger.info(f"保存面经: 新增 {new_count}（重复标题 {dup_count}）, 已存在 {len(existing)}")
        return new_count, existing, dup_count
    finally:
        conn.close()


def count_interviews(keyword_id: Optional[int] = None, include_irrelevant: bool = False) -> int:
    """统计面经总数（可按关键词筛选；默认排除标记为不相关的面经）"""
    conn = get_connection()
    try:
        where, params = _interview_where(keyword_id, include_irrelevant)
        row = conn.execute(f"SELECT COUNT(*) FROM interviews {where}", params).fetchone()
        return row[0]
    finally:
        conn.close()


def _interview_where(keyword_id: Optional[int], include_irrelevant: bool) -> tuple[str, list]:
    clauses = []
    params = []
    if keyword_id:
        clauses.append("keyword_id = ?")
        params.append(keyword_id)
    if not include_irrelevant:
        clauses.append("COALESCE(is_irrelevant, 0) = 0")
    return ("WHERE " + " AND ".join(clauses)) if clauses else "", params


def list_interviews(
    keyword_id: Optional[int] = None,
    limit: int = 100,
    offset: int = 0,
    include_irrelevant: bool = False,
) -> list[dict]:
    conn = get_connection()
    try:
        where, params = _interview_where(keyword_id, include_irrelevant)
        rows = conn.execute(
            f"SELECT * FROM interviews {where} ORDER BY publish_time DESC LIMIT ? OFFSET ?",
            (*params, limit, offset),
        ).fetchall()

        results = []
        for r in rows:
            item = dict(r)
            # 解析 tags JSON
            if item.get("tags"):
                try:
                    item["tags"] = json.loads(item["tags"])
                except (json.JSONDecodeError, TypeError):
                    item["tags"] = []
            # 获取问题列表
            qrows = conn.execute(
                "SELECT question FROM questions WHERE interview_id = ? ORDER BY id",
                (item["id"],)
            ).fetchall()
            item["questions"] = [q["question"] for q in qrows]
            # 截断 content 用于列表展示
            item["content_preview"] = item["content"][:200] + "..." if len(item["content"]) > 200 else item["content"]
            results.append(item)
        return results
    finally:
        conn.close()


def get_interview(id: int) -> Optional[dict]:
    conn = get_connection()
    try:
        row = conn.execute("SELECT * FROM interviews WHERE id = ?", (id,)).fetchone()
        if not row:
            return None
        item = dict(row)
        if item.get("tags"):
            try:
                item["tags"] = json.loads(item["tags"])
            except (json.JSONDecodeError, TypeError):
                item["tags"] = []
        qrows = conn.execute(
            "SELECT question FROM questions WHERE interview_id = ? ORDER BY id",
            (id,)
        ).fetchall()
        item["questions"] = [q["question"] for q in qrows]
        return item
    finally:
        conn.close()


def delete_interview(id: int):
    conn = get_connection()
    try:
        conn.execute("DELETE FROM questions WHERE interview_id = ?", (id,))
        conn.execute("DELETE FROM interviews WHERE id = ?", (id,))
        conn.commit()
    finally:
        conn.close()


# ========== 大模型清洗 ==========

def list_clean_targets(keyword_id: int) -> list[dict]:
    """
    取出某关键词下待清洗的面经：
    已清洗过（cleaned_at 非空）或已判定不相关的都跳过，保证增量数据只清洗一次。
    """
    conn = get_connection()
    try:
        rows = conn.execute(
            """SELECT id, title, content, keyword_id, COALESCE(is_irrelevant,0) AS is_irrelevant
               FROM interviews
               WHERE keyword_id = ?
                 AND COALESCE(is_irrelevant,0) = 0
                 AND cleaned_at IS NULL
               ORDER BY publish_time DESC""",
            (keyword_id,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def save_cleaned_interview(interview_id: int, cleaned_content: str, qa: list[dict]) -> None:
    """保存大模型清洗结果：原文备份到 raw_content，正文替换为结构化问答，问答列表重建。
    qa: [{"q": 问题, "a": 原回答}]；重新清洗会清空旧的 AI 答案。"""
    conn = get_connection()
    try:
        # 只备份第一次清洗前的原文，避免重复清洗覆盖掉真正的原始内容
        conn.execute(
            """UPDATE interviews
               SET raw_content = COALESCE(raw_content, content),
                   content = ?,
                   cleaned_at = ?,
                   answered_at = NULL,
                   is_irrelevant = 0
               WHERE id = ?""",
            (cleaned_content, datetime.now().isoformat(), interview_id),
        )
        conn.execute("DELETE FROM questions WHERE interview_id = ?", (interview_id,))
        conn.executemany(
            "INSERT INTO questions (interview_id, question, answer) VALUES (?, ?, ?)",
            [(interview_id, item["q"], item.get("a") or None) for item in qa],
        )
        conn.commit()
    finally:
        conn.close()


# ========== AI 答题 ==========

def get_qa_pairs(interview_id: int) -> list[dict]:
    """取某篇面经的结构化问答（按清洗时顺序），供 AI 答题与详情页对照展示"""
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT id, question, answer, ai_answer FROM questions WHERE interview_id = ? ORDER BY id",
            (interview_id,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def backfill_question_answers(interview_id: int, qa: list[dict]) -> int:
    """旧版清洗数据 questions.answer 为空：按顺序用正文解析出的原回答回填。
    仅当解析条数与 questions 行数完全一致时才回填，避免错位；返回回填行数。"""
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT id FROM questions WHERE interview_id = ? ORDER BY id",
            (interview_id,),
        ).fetchall()
        if len(rows) != len(qa):
            logger.info(
                "跳过旧答案回填：interview=%s 问题行数 %d 与解析条数 %d 不一致",
                interview_id, len(rows), len(qa),
            )
            return 0
        for row, item in zip(rows, qa):
            ans = (item.get("a") or "").strip() or None
            conn.execute("UPDATE questions SET answer = ? WHERE id = ?", (ans, row["id"]))
        conn.commit()
        return len(rows)
    finally:
        conn.close()


def list_answer_targets(keyword_id: int) -> list[dict]:
    """待 AI 答题的面经：已清洗、未判不相关、尚未答过（增量）"""
    conn = get_connection()
    try:
        rows = conn.execute(
            """SELECT id, title, content, position, keyword_id
               FROM interviews
               WHERE keyword_id = ?
                 AND COALESCE(is_irrelevant,0) = 0
                 AND cleaned_at IS NOT NULL
                 AND answered_at IS NULL
               ORDER BY publish_time DESC""",
            (keyword_id,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def save_ai_answers(interview_id: int, answers: list[dict]) -> int:
    """保存一篇面经的 AI 答案。
    answers: [{"question_id": int, "ai_answer": str}]，空串/None 表示该题跳过。
    仅更新当前仍属于该面经的题目（防止与「重新清洗」并发时 UPDATE 落空）；
    没有任何有效命中时不置 answered_at，返回 0，由调用方按失败处理。"""
    conn = get_connection()
    try:
        valid_ids = {
            r["id"] for r in conn.execute(
                "SELECT id FROM questions WHERE interview_id = ?", (interview_id,)
            ).fetchall()
        }
        items = [x for x in answers if x.get("question_id") in valid_ids]
        if not items:
            conn.rollback()
            return 0
        for item in items:
            text = (item.get("ai_answer") or "").strip() or None
            conn.execute(
                "UPDATE questions SET ai_answer = ? WHERE id = ? AND interview_id = ?",
                (text, item["question_id"], interview_id),
            )
        conn.execute(
            "UPDATE interviews SET answered_at = ? WHERE id = ?",
            (datetime.now().isoformat(), interview_id),
        )
        conn.commit()
        return len(items)
    finally:
        conn.close()


def mark_interview_irrelevant(interview_id: int, irrelevant: bool = True) -> None:
    """
    标记/取消标记「与关键词不相关」（软删除，不删数据）。
    标记时记 cleaned_at 避免重复调模型；
    恢复时清 cleaned_at，使其回到「待清洗」，下次批量可重新处理。
    """
    conn = get_connection()
    try:
        if irrelevant:
            conn.execute(
                """UPDATE interviews
                   SET is_irrelevant = 1, cleaned_at = COALESCE(cleaned_at, ?)
                   WHERE id = ?""",
                (datetime.now().isoformat(), interview_id),
            )
        else:
            conn.execute(
                "UPDATE interviews SET is_irrelevant = 0, cleaned_at = NULL WHERE id = ?",
                (interview_id,),
            )
        conn.commit()
    finally:
        conn.close()


def list_irrelevant(keyword_id: Optional[int] = None, limit: int = 20, offset: int = 0) -> list[dict]:
    """归档区：列出被判定不相关的面经（可按关键词筛选）"""
    conn = get_connection()
    try:
        where, params = ("WHERE i.is_irrelevant = 1 AND i.keyword_id = ?", [keyword_id]) \
            if keyword_id else ("WHERE i.is_irrelevant = 1", [])
        rows = conn.execute(
            f"""SELECT i.*, k.name AS keyword_name
                FROM interviews i LEFT JOIN keywords k ON k.id = i.keyword_id
                {where} ORDER BY i.cleaned_at DESC LIMIT ? OFFSET ?""",
            (*params, limit, offset),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def count_irrelevant(keyword_id: Optional[int] = None) -> int:
    conn = get_connection()
    try:
        if keyword_id:
            row = conn.execute(
                "SELECT COUNT(*) FROM interviews WHERE is_irrelevant = 1 AND keyword_id = ?",
                (keyword_id,),
            ).fetchone()
        else:
            row = conn.execute("SELECT COUNT(*) FROM interviews WHERE is_irrelevant = 1").fetchone()
        return row[0]
    finally:
        conn.close()


def get_clean_stats() -> dict:
    """清洗概况：总数 / 已清洗 / 已答题 / 不相关 / 待清洗"""
    conn = get_connection()
    try:
        row = conn.execute(
            """SELECT
                 COUNT(*) AS total,
                 SUM(CASE WHEN cleaned_at IS NOT NULL AND COALESCE(is_irrelevant,0) = 0 THEN 1 ELSE 0 END) AS cleaned,
                 SUM(CASE WHEN answered_at IS NOT NULL AND COALESCE(is_irrelevant,0) = 0 THEN 1 ELSE 0 END) AS answered,
                 SUM(COALESCE(is_irrelevant,0)) AS irrelevant,
                 SUM(CASE WHEN cleaned_at IS NULL AND COALESCE(is_irrelevant,0) = 0 THEN 1 ELSE 0 END) AS pending
               FROM interviews"""
        ).fetchone()
        return {
            "total": row["total"] or 0,
            "cleaned": row["cleaned"] or 0,
            "answered": row["answered"] or 0,
            "irrelevant": row["irrelevant"] or 0,
            "pending": row["pending"] or 0,
        }
    finally:
        conn.close()


def get_stats() -> dict:
    """获取总体统计"""
    conn = get_connection()
    try:
        keyword_count = conn.execute("SELECT COUNT(*) FROM keywords").fetchone()[0]
        interview_count = conn.execute(
            "SELECT COUNT(*) FROM interviews WHERE COALESCE(is_irrelevant,0) = 0"
        ).fetchone()[0]
        question_count = conn.execute(
            """SELECT COUNT(*) FROM questions q
               JOIN interviews i ON i.id = q.interview_id
               WHERE COALESCE(i.is_irrelevant,0) = 0"""
        ).fetchone()[0]
        latest = conn.execute(
            "SELECT MAX(publish_time) as latest FROM interviews"
        ).fetchone()["latest"]
        return {
            "keyword_count": keyword_count,
            "interview_count": interview_count,
            "question_count": question_count,
            "latest_interview": latest,
        }
    finally:
        conn.close()


# ========== 爬取任务日志 ==========

def save_crawl_task(
    task_id: str,
    query: str,
    keyword_id: Optional[int],
    days_filter: int,
    max_pages: int,
    max_posts: int,
    status: str,
    stats: dict,
    result: Optional[dict],
    steps: list,
    started_at: str,
    finished_at: Optional[str],
) -> int:
    """保存/更新爬取任务"""
    conn = get_connection()
    try:
        existing = conn.execute("SELECT id FROM crawl_tasks WHERE task_id = ?", (task_id,)).fetchone()
        if existing:
            conn.execute(
                """UPDATE crawl_tasks SET status=?, stats=?, result=?, steps=?, finished_at=?
                   WHERE task_id=?""",
                (status, json.dumps(stats, ensure_ascii=False),
                 json.dumps(result, ensure_ascii=False) if result else None,
                 json.dumps(steps, ensure_ascii=False), finished_at, task_id),
            )
            conn.commit()
            return existing["id"]
        else:
            cur = conn.execute(
                """INSERT INTO crawl_tasks
                   (task_id, query, keyword_id, days_filter, max_pages, max_posts,
                    status, stats, result, steps, started_at, finished_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                (task_id, query, keyword_id, days_filter, max_pages, max_posts,
                 status,
                 json.dumps(stats, ensure_ascii=False),
                 json.dumps(result, ensure_ascii=False) if result else None,
                 json.dumps(steps, ensure_ascii=False),
                 started_at, finished_at),
            )
            conn.commit()
            return cur.lastrowid
    finally:
        conn.close()


def list_crawl_tasks(limit: int = 50) -> list[dict]:
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT * FROM crawl_tasks ORDER BY started_at DESC LIMIT ?", (limit,)
        ).fetchall()
        result = []
        for r in rows:
            item = dict(r)
            for col in ("stats", "result", "steps"):
                if item.get(col):
                    try: item[col] = json.loads(item[col])
                    except Exception: pass
            if isinstance(item.get("steps"), list):
                item["steps_preview"] = item["steps"][:10]
                item["steps_count"] = len(item["steps"])
            result.append(item)
        return result
    finally:
        conn.close()


def get_crawl_task(task_id: str) -> Optional[dict]:
    conn = get_connection()
    try:
        row = conn.execute("SELECT * FROM crawl_tasks WHERE task_id = ?", (task_id,)).fetchone()
        if not row:
            return None
        item = dict(row)
        for col in ("stats", "result", "steps"):
            if item.get(col):
                try: item[col] = json.loads(item[col])
                except Exception: pass
        return item
    finally:
        conn.close()


def delete_crawl_task(task_id: str):
    conn = get_connection()
    try:
        conn.execute("DELETE FROM crawl_tasks WHERE task_id = ?", (task_id,))
        conn.commit()
    finally:
        conn.close()


def clear_all_crawl_tasks():
    conn = get_connection()
    try:
        conn.execute("DELETE FROM crawl_tasks")
        conn.commit()
    finally:
        conn.close()


# ========== AI 任务日志（清洗 / 答题） ==========

def save_ai_task(
    task_id: str,
    kind: str,
    keyword_name: str,
    status: str,
    stats: dict,
    steps: list,
    started_at: str,
    finished_at: Optional[str],
    keyword_id: Optional[int] = None,
) -> int:
    """保存/更新清洗或答题任务（kind: clean / answer）"""
    conn = get_connection()
    try:
        existing = conn.execute("SELECT id FROM ai_tasks WHERE task_id = ?", (task_id,)).fetchone()
        if existing:
            conn.execute(
                """UPDATE ai_tasks SET keyword_id=?, status=?, stats=?, steps=?, finished_at=?
                   WHERE task_id=?""",
                (keyword_id, status, json.dumps(stats, ensure_ascii=False),
                 json.dumps(steps, ensure_ascii=False), finished_at, task_id),
            )
            conn.commit()
            return existing["id"]
        cur = conn.execute(
            """INSERT INTO ai_tasks
               (task_id, kind, keyword_id, keyword_name, status, stats, steps, started_at, finished_at)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (task_id, kind, keyword_id, keyword_name, status,
             json.dumps(stats, ensure_ascii=False),
             json.dumps(steps, ensure_ascii=False),
             started_at, finished_at),
        )
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def _parse_ai_row(row) -> dict:
    item = dict(row)
    for col in ("stats", "steps"):
        if item.get(col):
            try:
                item[col] = json.loads(item[col])
            except Exception:
                pass
    if isinstance(item.get("steps"), list):
        item["steps_count"] = len(item["steps"])
    return item


def list_ai_tasks(kind: Optional[str] = None, limit: int = 100) -> list[dict]:
    conn = get_connection()
    try:
        if kind:
            rows = conn.execute(
                "SELECT * FROM ai_tasks WHERE kind = ? ORDER BY started_at DESC LIMIT ?",
                (kind, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM ai_tasks ORDER BY started_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [_parse_ai_row(r) for r in rows]
    finally:
        conn.close()


def get_ai_task(task_id: str) -> Optional[dict]:
    conn = get_connection()
    try:
        row = conn.execute("SELECT * FROM ai_tasks WHERE task_id = ?", (task_id,)).fetchone()
        return _parse_ai_row(row) if row else None
    finally:
        conn.close()


def delete_ai_task(task_id: str):
    conn = get_connection()
    try:
        conn.execute("DELETE FROM ai_tasks WHERE task_id = ?", (task_id,))
        conn.commit()
    finally:
        conn.close()


def clear_ai_tasks(kind: Optional[str] = None):
    """清空 AI 任务日志；kind 为空时清空全部清洗/答题记录"""
    conn = get_connection()
    try:
        if kind:
            conn.execute("DELETE FROM ai_tasks WHERE kind = ?", (kind,))
        else:
            conn.execute("DELETE FROM ai_tasks")
        conn.commit()
    finally:
        conn.close()


def mark_interrupted_ai_tasks() -> int:
    """
    服务启动时回收上次进程遗留的 running 记录。
    进程被强杀/重启时收尾的落库不会执行，这些任务永远停在 running，
    这里统一标记为 interrupted 并补一条说明，避免日志看起来「凭空消失」。
    """
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT task_id, kind, steps FROM ai_tasks WHERE status = 'running'"
        ).fetchall()
        if not rows:
            return 0
        now = datetime.now().isoformat()
        for r in rows:
            try:
                steps = json.loads(r["steps"] or "[]")
            except Exception:
                steps = []
            steps.append({
                "step": "interrupted", "phase": r["kind"] or "clean", "status": "skipped",
                "message": "⚠️ 任务因服务重启中断，以上为中断前已保存的日志",
                "current": 0, "total": 0, "extra": {},
            })
            conn.execute(
                "UPDATE ai_tasks SET status='interrupted', steps=?, finished_at=? WHERE task_id=?",
                (json.dumps(steps, ensure_ascii=False), now, r["task_id"]),
            )
        conn.commit()
        return len(rows)
    finally:
        conn.close()
