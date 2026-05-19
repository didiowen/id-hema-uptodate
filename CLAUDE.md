# CLAUDE.md — ID + Haematology Weekly Report

## Project Purpose

Auto-generate weekly Markdown reports on **Infectious Disease + Haematology (HSCT focus)** trends from:
- OpenEvidence MCP (`mcp__openevidence__oe_ask`)
- PubMed MCP (`mcp__claude_ai_PubMed__search_articles`)
- ClinicalTrials.gov MCP (`mcp__claude_ai_Clinical_Trials__search_trials`)
- CrossRef API (via `python main.py journals`) — CID + Blood by default
- Web news: CIDRAP AMR, CID RSS, EID (CDC), MMWR, ProMED, Puscast (MicrobeTV), ASH News

---

## Before Writing a New Report

**MANDATORY — do this BEFORE writing a single word of content:**

```bash
# 1. Find the latest report
PREV=$(ls reports/ -t | head -1)
echo "Previous report: $PREV"

# 2. Read it fully — note every outbreak, drug, trial, and section topic covered
# 3. Grep key programme / trial / pathogen names to see what's already documented
grep -E "letermovir|maribavir|cefiderocol|sulbactam-durlobactam|isavuconazole|posaconazole|HSCT|GVHD|CMV|EBV|BK virus|Candida auris|H5N1|mpox" reports/$PREV
```

After reading the previous report, answer these before writing:
- Which outbreaks / surveillance updates were already covered with the same numbers? → **skip entirely**
- Which trials had interim data last week? → include only if new follow-up published
- Which drug approvals / label expansions were already documented? → **skip unless indication / population changed**

**Do NOT repeat** any finding with identical numbers. Mark new follow-up data explicitly: `[更新]` before the subsection heading, and state what changed vs last week.

If a section has no genuinely new data this week: write `_本週無新訊號_` and move on.

---

## Report File Naming

```
reports/YYYY-WNN.md
```

Use ISO week number: `python3 -c "from datetime import date; d=date.today(); print(f'{d.year}-W{d.isocalendar()[1]:02d}')"`.

---

## Report Structure

### Required Sections (繁體中文)

```
# 感染症 + 血液腫瘤週報 — YYYY-WNN

> 生成日期：YYYY-MM-DD｜資料來源：...
> 涵蓋期間：...

---

## 摘要
（本週五大訊號 — bullet points, concrete numbers）

## 一、多重抗藥性革蘭氏陰性菌（CRE、CRAB、MDR-PA）
## 二、抗黴菌治療與 IFD（aspergillosis、mucormycosis、candidiasis）
## 三、CMV、EBV、BK virus 在 HSCT 受贈者
## 四、HSCT 與 GVHD 相關感染
## 五、AMR Stewardship 與新藥
## 六、Emerging Infectious Disease（H5N1、Mpox、zoonosis、One Health）
## 七、疫苗與預防（VZV、RSV、COVID 變異株）
## 八、進行中高優先試驗追蹤
## 九、台灣臨床情境備註
## 十、本週 Key Takeaways

## 十一、蜥蜴LLM 點評
（OpenEvidence分類：practice-changing vs hypothesis-generating）

## 十二、媒體動態
（CIDRAP / ProMED / EID / MMWR / ASH News table）

## 文獻速報 — CrossRef 期刊
（LLM-filtered CID + Blood articles）
```

Sections without new data this week should say: `_本週無新訊號_`

---

## Writing Style

- Language: **繁體中文**，英文術語保留原文（CMV, HSCT, GVHD, CRE, MIC, HR, OS, PFS 等）
- Every clinical claim must cite trial / study name + author + journal + DOI
- Tables: use Markdown tables for comparative data (drug vs comparator, MIC distribution, outbreak case-counts)
- Numbers: always include HR, CI, p-value when available; for outbreaks, case-counts + CFR
- Avoid vague superlatives; every "significant" needs a number

---

## Data Pipeline

Run in order before writing:

```bash
uv run python main.py scrape          # CIDRAP / CID / EID / MMWR / ProMED / Puscast / ASH News
uv run python main.py journals        # CID + Blood (CrossRef, keyword pre-screened)
```

For full pipeline (including Twitter if credentials available):

```bash
uv run python main.py run
```

Cached data locations:
- `data/webscrape_cache.json` — web news articles
- `data/journals_cache.json` — CrossRef journal articles (pre-screened, not yet final-filtered)

**CrossRef filtering note:** The Python fetcher applies a keyword pre-screen only (broad net).
When writing the report, read `data/journals_cache.json` and **filter in-session** — discard any
article whose primary topic falls outside ID / HSCT-related haematology (e.g. oncology articles
that share "transplant" terminology but aren't allo-HSCT or transplant-ID relevant; non-HSCT
solid-organ-transplant rejection trials; pure benign-haematology articles unrelated to infection).
Only include articles confirmed ID / HSCT-haem relevant in the `## 文獻速報` section.

---

## 蜥蜴LLM 點評 Section

Use `mcp__openevidence__oe_ask` with a prompt like:

```
Based on the following ID + Haematology (HSCT focus) findings from this week, classify each as:
- Practice-changing (changes standard of care NOW)
- Hypothesis-generating (promising but needs confirmation)
- Context-dependent (changes practice for specific subgroup only — e.g. high-risk CMV donor/recipient pairs, allo-HSCT only, neutropenic patients only)

[list findings with trial / study names and key numbers]
```

Extract result with: `result.extracted_answer_raw`

---

## After Writing

1. Check word count: report should be 3000–8000 words
2. Verify every table has header separators (`|---|---|`)
3. Run `uv run python main.py report` if auto-generating from DB
4. Commit: `git add reports/YYYY-WNN.md && git commit -m "report: YYYY-WNN"`
5. Push → GitHub Action auto-publishes to Wiki

---

## Duplicate-Avoidance Checklist

Before finalising, cross-check against the previous report:

```bash
PREV=$(ls reports/ -t | head -2 | tail -1)
# Check programme / trial / drug names
grep -E "letermovir|maribavir|cefiderocol|sulbactam-durlobactam|isavuconazole|posaconazole|HSCT|GVHD|CMV|EBV|BK virus|Candida auris|H5N1|mpox" reports/$PREV
# Check effect sizes — if same numbers appear, it's a repeat
grep -E "HR [0-9]|RR [0-9]|OR [0-9]|CFR [0-9]|MIC [0-9]|95% CI" reports/$PREV | head -20
```

Rules:
- Same study + same numbers → **delete the section**
- Same study + new data (updated follow-up, subgroup, approval) → keep with `[更新]` tag
- Brand new study / outbreak → include normally

---

## Switching to Another Domain

See `README.md` → "切換至其他領域" for step-by-step instructions on retargeting the YAML configs.
