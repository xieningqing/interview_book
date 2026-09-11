# 面经猎手（Interview Hunter）

牛客网面经采集与管理工具：按关键词自动搜索、抓取面经正文，本地归档为 SQLite + TXT，带可视化爬取进度与历史日志。

## 功能

- **关键词管理**：保存关键词及爬取参数（时间范围 / 搜索页数 / 条数），可重复爬取
- **自动采集**：边搜索边抓正文，自动翻页补够目标条数；正文过短、抓取失败自动跳过补位，支持手动停止（已抓数据保留）
- **面经库**：卡片浏览、按关键词筛选、查看详情，自动提取面试问题，URL/标题双重去重
- **爬取监控**：实时阶段流转、进度条、成功/失败/风控统计与请求日志
- **Cookie 管理**：粘贴浏览器 Cookie 即可，支持脱敏查看
- **IP 代理池**：支持 http / https / socks5，轮询/随机/固定策略，任务内会话粘性
- **反爬策略**：UA 轮换、随机延迟、指数退避重试、风控检测、SEO 降级页识别重试

## 技术栈

Python 3.10+ · FastAPI · httpx · Jinja2 · SQLite（标准库 sqlite3）· 原生 HTML/CSS/JS

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 启动服务
python run.py
```

浏览器打开 http://localhost:8000

## 使用流程

1. 打开 **Cookie** 页，从浏览器开发者工具复制牛客网的 Cookie 粘贴保存（未登录只能拿到很少的数据）
2. 在首页 **添加关键词**（如「agent 面经」），设置时间范围、搜索页数和目标条数
3. 点击 **爬取**，在进度面板实时观察；需要中断时点 **停止**，已抓到的面经会保留
4. 到 **面经库** 浏览、筛选、查看详情；TXT 同时保存在 `data/txt/关键词/` 目录
5. 高频抓取建议在 **代理** 页配置代理池，降低风控概率

## 目录结构

```
interview_book/
├── run.py                  # 启动入口（uvicorn，端口 8000）
├── requirements.txt
├── app/
│   ├── main.py             # FastAPI 路由与任务调度
│   ├── config.py           # 爬虫参数 / UA 池 / 阈值配置
│   ├── crawler/nowcoder.py # 牛客搜索、详情抓取、反爬逻辑
│   ├── storage/            # database / cookie / proxy / txt 存储
│   ├── templates/          # Jinja2 页面
│   └── static/style.css    # 全站样式
└── data/                   # 运行后生成（含敏感数据，已在 .gitignore）
    ├── cookies.json
    ├── proxies.json
    ├── interview.db
    └── txt/
```

## 说明

- 正文「有效长度」阈值默认 30 字符，可在 `app/config.py` 的 `min_content_length` 调整
- 补充条数受「搜索页数」上限约束，关键词结果普遍很短时请调大页数
- `data/cookies.json`、数据库等含个人数据，**请勿提交或分享**
