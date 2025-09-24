import io, json, time, math
from datetime import datetime, timedelta
import requests, pandas as pd, streamlit as st

st.set_page_config(page_title="AdAI 配分ビュー（CSV→要約→gpt-5）", layout="wide")

# ---------------------------
# セッション状態
# ---------------------------
if "raw" not in st.session_state:      st.session_state.raw = None     # n8n生JSON
if "df" not in st.session_state:       st.session_state.df = None      # wide DataFrame
if "features" not in st.session_state: st.session_state.features = None# LLM入力の要約JSON
if "result" not in st.session_state:   st.session_state.result = None  # LLMの結果JSON

# ---------------------------
# 取得（手動のみ / Bearer無し。BasicはSecretsがあれば自動使用）
# ---------------------------
def _http_get_latest(timeout_s: int = 20):
    url = st.secrets["N8N_JSON_URL"]
    auth = None
    if "N8N_BASIC_USER" in st.secrets and "N8N_BASIC_PASS" in st.secrets:
        auth = (st.secrets["N8N_BASIC_USER"], st.secrets["N8N_BASIC_PASS"])
    r = requests.get(url, auth=auth, timeout=timeout_s)
    r.raise_for_status()
    return r.json()

def fetch_latest_manual(force: bool = False):
    if force: st.cache_data.clear()
    timeouts = [10, 20, 40]
    last_err = None
    for i, t in enumerate(timeouts, start=1):
        try:
            return _http_get_latest(timeout_s=t)
        except requests.exceptions.ReadTimeout as e:
            last_err = e
            time.sleep(0.8 * i)
    raise last_err or requests.exceptions.ReadTimeout("n8n webhook timeout")

# ---------------------------
# CSV → DataFrame
# ---------------------------
def parse_wide_csv(csv_text: str) -> pd.DataFrame:
    df = pd.read_csv(io.StringIO(csv_text))
    if "variable" not in df.columns:
        raise ValueError("CSVに 'variable' 列がありません")
    df = df.set_index("variable")
    # 日付列を時系列順に
    cols = []
    for c in df.columns:
        try:
            cols.append((pd.to_datetime(c), c))
        except Exception:
            cols.append((None, c))
    # 日付として解釈できた列のみソートして先に、非日付は末尾
    date_cols = sorted([c for d, c in cols if d is not None])
    other_cols = [c for d, c in cols if d is None]
    df = df[date_cols + other_cols]
    # 数値化
    for c in date_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

# ---------------------------
# 直近7日要約の算出（facts + channels）
# ---------------------------
def last_n(series: pd.Series, n=7):
    # 後ろからn件の非NaN
    vals = series.dropna().iloc[-n:]
    return vals

def safe_median(series: pd.Series, n=7):
    vals = last_n(series, n)
    return float(vals.median()) if len(vals) else None

def safe_sum(series: pd.Series, n=7):
    vals = last_n(series, n)
    return float(vals.sum()) if len(vals) else 0.0

def build_features(df: pd.DataFrame, meta: dict):
    # meta から期間
    ads_min = pd.to_datetime(meta.get("adsMinDate"))
    ads_max = pd.to_datetime(meta.get("adsMaxDate"))
    if pd.isna(ads_max):
        # DataFrameの最後の日付列から推定
        try:
            ads_max = pd.to_datetime([c for c in df.columns if c[:4].isdigit()][-1])
        except Exception:
            ads_max = pd.Timestamp.today().normalize()
    target_date = (ads_max + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    month_start = ads_max.replace(day=1).strftime("%Y-%m-%d")
    month_end = (ads_max + pd.offsets.MonthEnd(0)).strftime("%Y-%m-%d")

    # 全体コスト系列
    cost_all = df.loc["コスト_ALL"].dropna()
    # MTD（今月分合計）
    m_mask = (pd.to_datetime(cost_all.index) >= pd.to_datetime(month_start)) & \
             (pd.to_datetime(cost_all.index) <= ads_max)
    mtd_spend = float(cost_all[m_mask].sum()) if len(cost_all[m_mask]) else 0.0

    # 参照候補
    yesterday_total = float(cost_all.iloc[-1]) if len(cost_all) else 0.0
    avg_last3 = float(last_n(cost_all, 3).mean()) if len(last_n(cost_all, 3)) else 0.0
    median_last7 = float(last_n(cost_all, 7).median()) if len(last_n(cost_all, 7)) else 0.0

    channels = {}
    for key in ["IGFB","Google","YT","Tik"]:
        def row(name):
            rname = f"{name}_{key}"
            return df.loc[rname] if rname in df.index else pd.Series(dtype=float)
        channels[key] = {
            "last7": {
                "median_CPA": safe_median(row("CPA")),
                "median_CVR": safe_median(row("CVR")),
                "median_CPC": safe_median(row("CPC")),
                "clicks_sum": int(round(safe_sum(row("クリック数")))),
                "cv_sum":     int(round(safe_sum(row("CV")))),
                "cost_sum":   float(round(safe_sum(row("コスト")), 3)),
                "days_used":  int(last_n(row("コスト")).count())
            }
        }

    features = {
        "facts": {
            "target_date": target_date,
            "yesterday_date": ads_max.strftime("%Y-%m-%d"),
            "month_start": month_start,
            "month_end": month_end,
            "mtd_spend": round(mtd_spend, 3),
            "baseline_today": 0,
            "candidates": {
                "baseline_today": 0,
                "yesterday_total": round(yesterday_total, 3),
                "avg_last3": round(avg_last3, 3),
                "median_last7": round(median_last7, 3)
            }
        },
        "channels": channels
    }
    return features

# ---------------------------
# LLM（OpenAI / gpt-5）
# ---------------------------
BASE_INSTRUCTIONS = """あなたは広告予算配分の最適化アシスタントです。出力は厳密なJSONオブジェクトのみで返してください（余分な文章・コードフェンス不可）。

【目的】
- 直近指標（last7中央値や合計）を踏まえ、CPA最適化の観点で「本日の総投資額」と「媒体別配分」を決定する。
- 月予算は未提供。総額は直近の水準（yesterday_total/avg_last3/median_last7等）を参照して提案してよい（抑制も可）。

【前提】
- 媒体ラベルは IGFB / Google / YT / Tik の4つのみ。
- 欠損は0換算しない。観測値のみで統計して良いが、不確実性は説明に反映。
- today_total_spend と各 amount は 整数JPY（四捨五入）。
- share は小数2桁。合計は1.00に丸めて最大share媒体で差分吸収。
- amount 合計は today_total_spend に厳密一致（丸め差は最大share媒体のamountで吸収）。

【出力スキーマ（厳密）】
{
  "today_total_spend": <int>,
  "allocation": {
    "IGFB":   { "share": <number>, "amount": <int> },
    "Google": { "share": <number>, "amount": <int> },
    "YT":     { "share": <number>, "amount": <int> },
    "Tik":    { "share": <number>, "amount": <int> }
  },
  "reasoning_points": ["…","…"],
  "report": {
    "title": "本日の予算配分レポート",
    "target_date": "<YYYY-MM-DD>",
    "executive_summary": "一段落で要旨",
    "pacing_decision": { "chosen_today_total": <int>, "why": "数値根拠を説明" },
    "channel_signals": {
      "IGFB":  { "stance": "積極/中庸/抑制", "last7": { "median_CPA": <number|null>, "median_CVR": <number|null>, "median_CPC": <number|null>, "clicks_sum": <int>, "cv_sum": <int> }, "confidence": "<low|med|high>", "why": "…" },
      "Google":{ "stance": "積極/中庸/抑制", "last7": { "median_CPA": <number|null>, "median_CVR": <number|null>, "median_CPC": <number|null>, "clicks_sum": <int>, "cv_sum": <int> }, "confidence": "<low|med|high>", "why": "…" },
      "YT":    { "stance": "積極/中庸/抑制", "last7": { "median_CPA": <number|null>, "median_CVR": <number|null>, "median_CPC": <number|null>, "clicks_sum": <int>, "cv_sum": <int> }, "confidence": "<low|med|high>", "why": "…" },
      "Tik":   { "stance": "積極/中庸/抑制", "last7": { "median_CPA": <number|null>, "median_CVR": <number|null>, "median_CPC": <number|null>, "clicks_sum": <int>, "cv_sum": <int> }, "confidence": "<low|med|high>", "why": "…" }
    },
    "decision_flow": [
      { "step": "総額の決定", "data_used": ["baseline_today","yesterday_total","avg_last3","median_last7"], "logic": "重み付けの考え方" },
      { "step": "配分の決定", "data_used": ["channel_signals","ボリューム","効率の相対比較"], "logic": "シェア比の導出" },
      { "step": "整合調整", "data_used": ["丸め誤差"], "logic": "最大share媒体で吸収" }
    ],
    "final_allocation_check": {
      "expected_cv": { "IGFB": <number|null>, "Google": <number|null>, "YT": <number|null>, "Tik": <number|null> },
      "notes": "CPA>0 時は expected_cv=金額/median_CPA。欠損時は null。"
    },
    "risks_and_actions": { "risks": ["…"], "next_actions": ["…"] }
  }
}

【入力（要約JSON）】
"""

def build_llm_input(features: dict) -> str:
    return BASE_INSTRUCTIONS + json.dumps(features, ensure_ascii=False, separators=(",", ":"))

def call_openai_chat(prompt: str) -> str:
    api_key = st.secrets["OPENAI_API_KEY"]
    base_url = st.secrets.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    model = st.secrets.get("OPENAI_MODEL", "gpt-5")
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are an expert ads budget allocation assistant. Output valid JSON only."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2,
    }
    r = requests.post(f"{base_url}/chat/completions",
                      headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                      json=body, timeout=120)
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"]

def strip_code_fence(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        parts = s.split("```")
        if len(parts) >= 3:
            s = parts[1]
        s = s.replace("json", "", 1).strip()
    return s.strip("` \n\r\t")

def enforce_constraints(obj: dict) -> dict:
    medias = ["IGFB","Google","YT","Tik"]
    total = max(0, int(obj.get("today_total_spend", 0)))
    alloc = obj.get("allocation", {}) or {}
    for m in medias:
        if m not in alloc: alloc[m] = {"share": 0, "amount": 0}
        alloc[m]["share"] = round(float(alloc[m].get("share", 0) or 0), 2)
    ssum = round(sum(alloc[m]["share"] for m in medias), 2)
    if ssum == 0 and total > 0:
        for m in medias: alloc[m]["share"] = 0.25
        ssum = 1.0
    diff = round(1.00 - ssum, 2)
    if abs(diff) >= 0.01:
        maxm = max(medias, key=lambda m: alloc[m]["share"])
        alloc[maxm]["share"] = round(alloc[maxm]["share"] + diff, 2)
    amts = [int(round(total * alloc[m]["share"])) for m in medias]
    adiff = total - sum(amts)
    if adiff != 0:
        maxm = max(medias, key=lambda m: alloc[m]["share"])
        amts[medias.index(maxm)] += adiff
    for i, m in enumerate(medias):
        alloc[m]["amount"] = max(0, int(amts[i]))
    obj["today_total_spend"] = total
    obj["allocation"] = alloc
    return obj

# ---------------------------
# UI
# ---------------------------
st.title("AdAI 配分ビュー（手動取得 → 要約 → gpt-5）")

with st.expander("手順", expanded=True):
    st.markdown("1) **データ取得**: n8n から CSV（wide）を取得 → DataFrame化\n"
                "2) **要約生成**: 直近7日の中央値/合計などを自動算出（facts/channels）\n"
                "3) **推論を実行**: gpt-5 へ要約JSONを渡し、最終配分JSONを生成")

c1, c2, c3 = st.columns([1,1,3])

with c1:
    if st.button("データ取得", type="primary"):
        with st.spinner("n8n から取得中…"):
            try:
                raw = fetch_latest_manual(False)
                st.session_state.raw = raw
                csv_text = (raw.get("csv", {}) or {}).get("wide", "")
                meta = raw.get("meta", {}) or {}
                if not csv_text:
                    st.error("csv.wide が空です")
                else:
                    df = parse_wide_csv(csv_text)
                    st.session_state.df = df
                    st.session_state.features = build_features(df, meta)
                    st.session_state.result = None
                    st.success("取得・要約成功")
            except Exception as e:
                st.error(f"取得に失敗: {e}")

with c2:
    if st.button("再取得（強制）"):
        with st.spinner("キャッシュ無視で再取得…"):
            try:
                raw = fetch_latest_manual(True)
                st.session_state.raw = raw
                csv_text = (raw.get("csv", {}) or {}).get("wide", "")
                meta = raw.get("meta", {}) or {}
                df = parse_wide_csv(csv_text)
                st.session_state.df = df
                st.session_state.features = build_features(df, meta)
                st.session_state.result = None
                st.success("再取得・要約成功")
            except Exception as e:
                st.error(f"再取得に失敗: {e}")

# プレビュー
if st.session_state.df is not None:
    st.subheader("CSVプレビュー（先頭数行）")
    st.dataframe(st.session_state.df.iloc[:10, -7:], use_container_width=True)
    st.subheader("要約（LLM入力）")
    st.code(json.dumps(st.session_state.features, ensure_ascii=False, indent=2))
else:
    st.info("まだデータ未取得です。**データ取得** を押してください。")

# 推論
st.markdown("### 推論（OpenAI: gpt-5）")
if "OPENAI_API_KEY" not in st.secrets:
    st.warning("Secrets に OPENAI_API_KEY を設定してください。")
else:
    disabled = st.session_state.features is None
    if st.button("推論を実行", disabled=disabled):
        if disabled:
            st.warning("先にデータ取得を行ってください。")
        else:
            with st.spinner("gpt-5 で推論中…"):
                try:
                    prompt = build_llm_input(st.session_state.features)
                    out = call_openai_chat(prompt)
                    out = strip_code_fence(out)
                    result = json.loads(out)
                    result = enforce_constraints(result)
                    st.session_state.result = result
                    st.success("推論成功")
                except json.JSONDecodeError as e:
                    st.error(f"LLM出力がJSONとして解釈できませんでした: {e}")
                    st.text(out[:1000] if isinstance(out, str) else str(out))
                except requests.exceptions.HTTPError as e:
                    st.error(f"OpenAI API エラー: {e}")
                except Exception as e:
                    st.error(f"推論に失敗: {e}")

# 結果表示
res = st.session_state.result
if res:
    td = res.get("report", {}).get("target_date") or st.session_state.features["facts"]["target_date"]
    st.subheader(f"本日の配分（{td}）")
    st.metric("総額", f"¥{int(res.get('today_total_spend', 0)):,}")
    alloc = res.get("allocation", {})
    df_view = pd.DataFrame([
        {"media":"IGFB",  "share(%)": round(alloc["IGFB"]["share"]*100,1),  "amount(¥)": alloc["IGFB"]["amount"]},
        {"media":"Google","share(%)": round(alloc["Google"]["share"]*100,1),"amount(¥)": alloc["Google"]["amount"]},
        {"media":"YT",    "share(%)": round(alloc["YT"]["share"]*100,   1),"amount(¥)": alloc["YT"]["amount"]},
        {"media":"Tik",   "share(%)": round(alloc["Tik"]["share"]*100,  1),"amount(¥)": alloc["Tik"]["amount"]},
    ])
    st.dataframe(df_view, use_container_width=True)

    if res.get("reasoning_points"):
        st.markdown("#### 判断ポイント")
        for p in res["reasoning_points"]:
            st.write("• " + str(p))
    if res.get("report"):
        st.markdown("#### 要約")
        st.write(res["report"].get("executive_summary",""))

    st.download_button("結果JSONをダウンロード",
                       data=json.dumps(res, ensure_ascii=False, indent=2),
                       file_name="allocation_result.json",
                       mime="application/json")

st.caption("※ CSV（wide）から必要指標をPythonで要約→gpt-5に渡して配分JSONを生成します。")
