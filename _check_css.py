"""临时校验脚本：检查模板中用到的 class 是否都在 CSS 中定义"""
import re
import pathlib

tpl_dir = pathlib.Path("app/templates")
css = pathlib.Path("app/static/style.css").read_text(encoding="utf-8")

used = set()
for f in tpl_dir.glob("*.html"):
    text = f.read_text(encoding="utf-8")
    for attr in re.findall(r'class="([^"]+)"', text):
        for c in attr.split():
            if "{" in c or "}" in c or c.startswith("%"):
                continue
            used.add(c)
    for attr in re.findall(r"className = ['\"]([^'\"]+)['\"]", text):
        for c in attr.split():
            if "{" in c or "$" in c:
                continue
            used.add(c)
    for attr in re.findall(r"classList\.(?:add|remove)\(['\"]([^'\"]+)['\"]", text):
        used.add(attr)

defined = set(re.findall(r"\.([a-zA-Z][\w-]*)", css))
missing = sorted(used - defined)

print("CSS 定义类数 :", len(defined))
print("模板使用类数 :", len(used))
print("缺失定义     :", missing if missing else "无")

# 检查 SVG symbol 引用是否都有定义
symbols = set(re.findall(r'<symbol id="([^"]+)"', (tpl_dir / "base.html").read_text(encoding="utf-8")))
refs = set()
for f in tpl_dir.glob("*.html"):
    refs |= set(re.findall(r'href="#(i-[\w-]+)"', f.read_text(encoding="utf-8")))
    refs |= set(re.findall(r"'(#i-[\w-]+)'", f.read_text(encoding="utf-8")))
    refs |= set(re.findall(r'"(#i-[\w-]+)"', f.read_text(encoding="utf-8")))
bad_refs = sorted(r.lstrip("#") for r in refs if r.lstrip("#") not in symbols)
print("symbol 定义  :", len(symbols))
print("未定义引用   :", bad_refs if bad_refs else "无")
