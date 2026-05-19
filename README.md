# ID + Haematology Weekly Trend Reporter

自動生成感染症 + 血液腫瘤每週趨勢報告，以繁體中文撰寫（英文醫學名詞不翻譯）。

目前設定：**感染症 + 血液腫瘤（Infectious Disease + Haematology, HSCT 重點）**

資料來源：OpenEvidence AI · PubMed · CIDRAP · ProMED · CDC EID & MMWR · CID · Puscast · ASH News · ClinicalTrials.gov

---

## 致謝 / Attribution

This project is a derivative work of [`htlin222/breast-cancer-uptodate`](https://github.com/htlin222/breast-cancer-uptodate). The original framework — YAML-driven pipeline, scraper architecture (`webscraper.py`, `crossref_fetcher.py`, `fetcher.py`), report generator, and wiki-publish GitHub Action — is by **htlin222**, published with MIT intent per the upstream README. This fork repurposes the codebase for Infectious Disease and Haematology (HSCT focus). See `LICENSE` for the full notice.

---

## 快速開始

```bash
# 安裝 uv（Python 套件管理）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 安裝相依套件
uv sync

# 執行爬蟲 + 報告（OncDaily / OncLive / ESMO）
uv run python main.py scrape

# 生成報告（需手動補充 OpenEvidence 段落，或直接在 Claude Code 中執行）
uv run python main.py report
```

報告輸出至 `reports/YYYY-Wxx.md`，push 到 `main` 後 GitHub Actions 自動發布至 Wiki。

---

## 專案結構

```
.
├── source/                   ← 所有可調整參數（不需改 Python）
│   ├── keywords.yml          ← 疾病相關關鍵詞（過濾用）
│   ├── drug_groups.yml       ← 藥物分組 + 會議關鍵詞
│   ├── search_queries.yml    ← Twitter 搜尋 query
│   ├── web_sources.yml       ← 爬蟲來源（RSS URL、Google News site）
│   └── twitter.yml           ← GraphQL op_id、cookie skip 清單
├── config/
│   └── seeds.txt             ← KOL Twitter 帳號種子清單
├── src/
│   ├── config.py             ← YAML 載入器（lru_cache）
│   ├── webscraper.py         ← 網路爬蟲（driven by web_sources.yml）
│   ├── fetcher.py            ← Twitter 爬蟲（driven by twitter.yml）
│   ├── reporter.py           ← 推文聚合報告生成
│   ├── discover.py           ← KOL 自動發掘
│   └── db.py                 ← SQLite 儲存
├── reports/                  ← 產出的週報（push 即觸發 wiki 發布）
├── main.py                   ← CLI 入口
└── .github/workflows/
    └── publish-wiki.yml      ← 自動發布 wiki 的 GitHub Action
```

---

## 切換至其他領域

本系統設計為**領域無關（domain-agnostic）**，所有領域知識都集中在 `source/` 下的 YAML 檔案，切換領域只需修改這幾個檔案，**不需動任何 Python 程式碼**。

> 以下以「切換至兒科 ID」為示範。其他領域（純 AMR、移植 ID、HIV、TB 等）做法相同。

### 步驟 1 — 替換 `source/keywords.yml`

```yaml
tid_eid_amr_keywords:       # ← 鍵名沿用即可（不影響功能）
  - paediatric infection
  - neonatal sepsis
  - RSV
  - influenza in children
  - measles
  - pertussis
  - vaccine-preventable
  - acute otitis media
  - bacterial meningitis
  - hand foot mouth disease
  - HFMD
  - enterovirus 71
  - EV71
```

### 步驟 2 — 替換 `source/drug_groups.yml`

依新領域重寫藥物分組：

```yaml
drug_groups:
  Paediatric_Antivirals:
    - oseltamivir
    - zanamivir
    - baloxavir
    - nirsevimab          # RSV passive immunisation
    - palivizumab
  Paediatric_Vaccines:
    - MMR
    - DTaP
    - HPV
    - rotavirus vaccine
    - pneumococcal conjugate

conference_keywords:
  - ESPID
  - PAS
  - IDWeek
  - WSPID
  - abstract
  - "#ESPID"
```

### 步驟 3 — 替換 `source/search_queries.yml`

```yaml
search_queries:
  - "(RSV OR bronchiolitis) (infant OR neonate OR pediatric)"
  - "(measles OR pertussis) outbreak"
  - "(EV71 OR enterovirus 71 OR HFMD) (Taiwan OR Asia)"
  - "(neonatal sepsis) (GBS OR antimicrobial)"
  - "(ESPID OR PAS) (vaccine OR pediatric)"
```

### 步驟 4 — 調整 `source/web_sources.yml`（選擇性）

大部分來源（CIDRAP、EID、MMWR、ProMED）已涵蓋廣泛 ID 主題；只需在 Google News 來源加 / 改 `query` 欄位：

```yaml
sources:
  - name: AAP News
    type: google_news
    domain: aap.org
    query: "infectious disease OR vaccine OR RSV OR pertussis"
    max_items: 20
    noise_filter: "membership|about|contact"
```

`webscraper.py` 會優先讀取 `query` 欄位，預設值為 `"infectious disease OR hematology"`。

### 步驟 5 — 替換 `config/seeds.txt`

```
# Paediatric ID KOLs
SpinalConvert    # Krow Ampofo (PIDS)
ESPIDSociety
PIDSociety
nadiashaman      # Nadia Shaman (pediatric ID)
```

### 步驟 6 — 改報告標題（選擇性）

`main.py` 的 `cmd_scrape` / `cmd_journals` 與 `src/reporter.py` 中的標題字串可直接改。

---

## 常見維護任務

| 問題 | 解法 |
|------|------|
| Twitter API 回 404 | 更新 `source/twitter.yml` 的 `op_id` |
| 某新藥沒被捕捉 | 加進 `source/keywords.yml` 和對應 `drug_groups.yml` |
| 新增爬蟲來源 | 在 `source/web_sources.yml` 加一筆 `type: rss` 或 `type: google_news` |
| 查詢字串太雜 | 修改 `source/search_queries.yml` |
| Wiki 沒更新 | 確認 push 路徑含 `reports/*.md`；或手動跑 Actions → `workflow_dispatch` |

---

## GitHub Actions — Wiki 自動發布

每次 push 包含 `reports/*.md` 的 commit 到 `main`，`.github/workflows/publish-wiki.yml` 自動：

1. 把新報告複製到 wiki repo
2. 重建 `Home.md` 索引（最新在前）
3. Force-push 到 wiki `master` branch

手動觸發：Actions → **Publish Reports to Wiki** → **Run workflow**

---

## 授權

MIT — 詳見 `LICENSE`。

原始框架版權屬 [`htlin222`](https://github.com/htlin222)，本 fork 的修改部分版權屬 didiowen。上游 repo 的 README 聲明 MIT，但目前尚未在 repo 內附 `LICENSE` 檔；建議至上游提 issue 請求補上以釐清授權狀態。
