"""
TXT 文件存储
按关键词组织目录，每篇面经存一个 txt 文件
"""
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

from app.config import TXT_DIR

logger = logging.getLogger(__name__)


def _safe_filename(name: str, max_len: int = 80) -> str:
    """清理文件名中的非法字符"""
    name = re.sub(r'[\\/:*?"<>|\n\r\t]', '_', name)
    name = name.strip().strip('.')
    return name[:max_len] if name else "untitled"


def _ensure_keyword_dir(keyword: str) -> Path:
    """确保关键词目录存在"""
    kw_dir = TXT_DIR / _safe_filename(keyword, 50)
    kw_dir.mkdir(parents=True, exist_ok=True)
    return kw_dir


def save_interview_to_txt(interview: dict, keyword: str) -> Path:
    """
    将单篇面经保存为 TXT 文件

    文件格式:
        ========================================
        标题: xxx
        作者: xxx
        来源: xxx
        链接: https://...
        发布时间: 2024-01-01
        关键词: xxx
        学校: xxx
        职位: xxx
        标签: tag1, tag2
        ----------------------------------------
        面试问题:
        1. xxx
        2. xxx
        ----------------------------------------
        正文内容:
        ========================================
        ...正文...
    """
    kw_dir = _ensure_keyword_dir(keyword)

    # 文件名: 日期_标题.txt
    pub_time = interview.get("publish_time")
    date_str = pub_time.strftime("%Y-%m-%d") if isinstance(pub_time, datetime) else "unknown_date"
    title = interview.get("title", "untitled")
    filename = f"{date_str}_{_safe_filename(title)}.txt"

    # 处理可能重名
    file_path = kw_dir / filename
    counter = 1
    while file_path.exists():
        file_path = kw_dir / f"{date_str}_{_safe_filename(title)}_{counter}.txt"
        counter += 1

    # 构建文件内容
    lines = []
    lines.append("=" * 60)
    lines.append(f"标题: {interview.get('title', '')}")
    lines.append(f"作者: {interview.get('author', '')}")
    lines.append(f"来源: {interview.get('source', 'nowcoder')}")
    lines.append(f"链接: {interview.get('url', '')}")
    if pub_time:
        time_str = pub_time.strftime("%Y-%m-%d %H:%M:%S") if isinstance(pub_time, datetime) else str(pub_time)
        lines.append(f"发布时间: {time_str}")
    else:
        lines.append("发布时间: 未知")
    lines.append(f"关键词: {keyword}")
    lines.append(f"学校: {interview.get('school', '')}")
    lines.append(f"职位: {interview.get('position', '')}")
    tags = interview.get("tags", [])
    lines.append(f"标签: {', '.join(tags) if tags else ''}")
    lines.append("-" * 60)

    # 面试问题列表
    questions = interview.get("questions", [])
    if questions:
        lines.append("面试问题:")
        for i, q in enumerate(questions, 1):
            lines.append(f"  {i}. {q}")
        lines.append("-" * 60)

    lines.append("正文内容:")
    lines.append("=" * 60)
    lines.append(interview.get("content", ""))

    file_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"  📄 已保存: {file_path.name}")
    return file_path


def save_batch_to_txt(interviews: list[dict], keyword: str) -> list[Path]:
    """批量保存，返回保存的文件路径列表"""
    paths = []
    for inv in interviews:
        try:
            p = save_interview_to_txt(inv, keyword)
            paths.append(p)
        except Exception as e:
            logger.warning(f"保存 txt 失败 {inv.get('title', '')}: {e}")
    return paths


def list_txt_files(keyword: Optional[str] = None) -> list[Path]:
    """列出 txt 文件"""
    if keyword:
        kw_dir = TXT_DIR / _safe_filename(keyword, 50)
        if kw_dir.exists():
            return sorted(kw_dir.glob("*.txt"), reverse=True)
        return []
    return sorted(TXT_DIR.rglob("*.txt"), reverse=True)


def get_txt_content(path: str) -> Optional[str]:
    """读取 txt 文件内容"""
    p = Path(path)
    if p.exists():
        return p.read_text(encoding="utf-8")
    return None
