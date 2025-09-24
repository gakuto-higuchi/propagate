# app.py
# AdAI 配分ビュー（CSV/JSON → features正規化 → プロンプト編集 → OpenAI推論 A/B/C 比較）
# - n8nの返却: ①{"csv":{"wide":...},"meta":...} / ②{"facts":...} / ③[{"facts":...},{"csv":{"wide":...},"meta":...}]
# - OpenAI: 公式SDK（openai>=1.0）を優先し、無ければHTTPフォールバック
# - 追加UI: 月予算・対象日の上書き、CSV同梱の有無/最新N日、A/B/Cのプロンプト編集＆比較、最終プロンプト＆生出力の表示

import io, json, time
from datetime import datetime
from zoneinfo import ZoneInfo

import requests
import pandas as pd
import streamlit as st

st.set_page_config(page_title="AdAI 配分ビュー（CSV/JSON→要約→OpenAI A/B/C）", layout="wide")

# ---------------------------
# セッション状態
# ---------------------------
if "raw" not in st.session_state:      st.session_state.raw = None          # n8n生
if "df" not in st.session_state:       st.session_state.df = None           # wide DataFrame
if "features" not in st.session_state: st.session_state.features = None     # LLM入力
if "csv_raw_text" not in st.session_state: st.session_state.csv_raw_text = None

# プロンプト/結果（A/B/C）
for key in ["A", "B", "C"]:
    if f"prompt_{key}" not in st.session_state:
        st.session_state[f"prompt_{key}"] = None
    if f"last_prompt_{key}" not in st.session_state:
        st.session_state[f"last_prompt_{key}"] = None
    if f"raw_output_{key}" not in st.session_state:
        st.session_state[f"raw_output_{key}"] = None
    if f"result_{key}" not in st.session_state:
        st.session_state[f"result_{key}"] = None

# CSV→プロンプト同梱制御
if "include_csv_in_prompt" not in st.session_state:
    st.session_state.include_csv_in_prompt = True
if "csv_prompt_days" not in st.session_state:
    st.session_state.csv_prompt_days = 14  # 最新N日だけ同梱

# オーバーライド
if "month_budget_override" not in st.session_state:
    st.session_state.month_budget_override = None
if "target_date_override" not in st.session_state:
    st.session_state.target_date_override = None

# ---------------------------
# n8n取得（手動のみ）
# ---------------------------
def _http_get_latest(timeout_s: int = 20):
    url = st.secrets["N8N_JSON_URL"]  # 例: https://<your-n8n>/webhook/latest
    auth = None
    if "N8N_BASIC_USER" in st.secrets and "N8N_BASIC_PASS" in st.secrets:
        auth = (st.secrets["N8N_BASIC_USER"], st.secrets["N8N_BASIC_PASS"])
    r = requests.get(url, auth=auth, timeout=timeout_s)
    r.raise_for_status()
    return r.json()

def fetch_latest_manual(force: bool = False):
    if force:
        st.cache_data.clear()
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
# 入力正規化
# ---------------------------
def parse_wide_csv(csv_text: str) -> pd.DataFrame:
    df = pd.read_csv(io.StringIO(csv_text))
    if "variable" not in df.columns:
        raise ValueError("CSVに 'variable' 列がありません")
    df = df.set_index("variable")

    # 日付列を時系列順に、非日付は末尾へ
    cols = []
    for c in df.columns:
        try:
            cols.append((pd.to_datetime(c), c))
        except Exception:
            cols.append((None, c))
    date_cols = sorted([c for d, c in cols if d is not None])
    other_cols = [c for d, c in cols if d is None]
    df = df[date_cols + other_cols]
    for c in date_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def last_n(series: pd.Series, n=7):
    return series.dropna().iloc[-n:]

def safe_median(series: pd.Series, n=7):
    vals = last_n(series, n)
    return float(vals.median()) if len(vals) else None

def safe_sum(series: pd.Series, n=7):
    vals = last_n(series, n)
    return float(vals.sum()) if len(vals) else 0.0

def build_features_from_csv(df: pd.DataFrame, meta: dict):
    # 期間推定
    ads_max = pd.to_datetime((meta or {}).get("adsMaxDate")) if meta else None
    if ads_max is None or pd.isna(ads_max):
        try:
            date_like = [c for c in df.columns if str(c)[:4].isdigit()]
            ads_max = pd.to_datetime(date_like[-1])
        except Exception:
            ads_max = pd.Timestamp.today().normalize()

    target_date = (ads_max + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    month_start = ads_max.replace(day=1).strftime("%Y-%m-%d")
    month_end = (ads_max + pd.offsets.MonthEnd(0)).strftime("%Y-%m-%d")

    # 全体コスト系列
    if "コスト_ALL" in df.index:
        cost_all = df.loc["コスト_ALL"].dropna()
    else:
        cost_all = pd.Series(dtype=float)

    if len(cost_all):
        idx_as_dt = pd.to_datetime(cost_all.index)
        m_mask = (idx_as_dt >= pd.to_datetime(month_start)) & (idx_as_dt <= ads_max)
        mtd_spend = float(cost_all[m_mask].sum()) if m_mask.any() else 0.0
        yesterday_total = float(cost_all.iloc[-1])
        avg_last3 = float(last_n(cost_all, 3).mean()) if len(last_n(cost_all, 3)) else 0.0
        median_last7 = float(last_n(cost_all, 7).median()) if len(last_n(cost_all, 7)) else 0.0
    else:
        mtd_spend = yesterday_total = avg_last3 = median_last7 = 0.0

    channels = {}
    for key in ["IGFB", "Google", "YT", "Tik"]:
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

    return {
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

def build_features_from_factsish(obj):
    """obj が {"facts": {...}} or [{"facts": {...}}] の形を features に揃える"""
    if isinstance(obj, list):
        if not obj:
            raise ValueError("空リストを受信しました")
        obj = obj[0]

    if not isinstance(obj, dict) or "facts" not in obj:
        raise ValueError("facts を含むオブジェクトではありません")

    facts = dict(obj["facts"])  # copy
    # channels の場所：facts.channels or obj.channels
    channels = {}
    if "channels" in facts and isinstance(facts["channels"], dict):
        channels = facts["channels"]
        facts = {k: v for k, v in facts.items() if k != "channels"}
    elif "channels" in obj and isinstance(obj["channels"], dict):
        channels = obj["channels"]

    return {"facts": facts, "channels": channels}

def coerce_features_and_df(raw):
    """
    n8nの返却を判定して features / df を返す
    - Case A: {"csv":{"wide":...},"meta":...}
    - Case B: {"facts":...} or [{"facts":...}]
    - Case C: [{"facts":...}, {"csv":{"wide":...},"meta":...}]  ← 推奨
    返り値: (features, df)
    """
    df = None
    features = None
    csv_raw_text = None

    # 配列: factsとcsvをマージ
    if isinstance(raw, list):
        facts_obj = next((x for x in raw if isinstance(x, dict) and "facts" in x), None)
        csv_obj   = next((x for x in raw if isinstance(x, dict) and "csv" in x), None)

        if csv_obj and isinstance(csv_obj.get("csv"), dict) and "wide" in csv_obj["csv"]:
            csv_raw_text = csv_obj["csv"]["wide"]
            df = parse_wide_csv(csv_raw_text)

        if facts_obj:
            features = build_features_from_factsish(facts_obj)
        elif df is not None:
            features = build_features_from_csv(df, (csv_obj or {}).get("meta", {}) or {})

        if features is None and df is None:
            raise ValueError("配列内に facts / csv が見つかりませんでした")

        st.session_state.csv_raw_text = csv_raw_text
        return features, df

    # 単体facts
    if isinstance(raw, dict) and "facts" in raw:
        features = build_features_from_factsish(raw)
        return features, df

    # 単体csv
    if isinstance(raw, dict) and isinstance(raw.get("csv"), dict) and "wide" in raw["csv"]:
        csv_raw_text = raw["csv"]["wide"]
        df = parse_wide_csv(csv_raw_text)
        features = build_features_from_csv(df, raw.get("meta", {}) or {})
        st.session_state.csv_raw_text = csv_raw_text
        return features, df

    raise ValueError("未対応のペイロード形式です")

# ---------------------------
# LLM（OpenAI / SDK優先 → HTTPフォールバック）
# ---------------------------
BASE_INSTRUCTIONS = """あなたは広告予算配分の最適化アシスタントです。出力は厳密なJSONオブジェクトのみで返してください（余分な文章・コードフェンス不可）。

【目的】
- 直近指標（last7中央値や合計）など入力の facts/channels を踏まえ、CPA最適化の観点で「本日の総投資額」と「媒体別配分」を決定する。
- 月予算情報（あれば）や candidates（yesterday_total/avg_last3/median_last7等）は参照（抑制も可）。

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

【入力（features: facts/channels）】
"""

def slice_csv_for_prompt(df: pd.DataFrame, days: int) -> str | None:
    if df is None or df.empty:
        return None
    date_cols = [c for c in df.columns if str(c)[:4].isdigit()]
    if not date_cols:
        return None
    cols = date_cols[-max(1, int(days)):]  # 最低1日
    out = df.reset_index()[["variable"] + cols]
    return out.to_csv(index=False, float_format="%.3f")

def compose_user_prompt(base_prompt: str, features: dict, df: pd.DataFrame) -> str:
    uprompt = (base_prompt or BASE_INSTRUCTIONS) + json.dumps(features, ensure_ascii=False, separators=(",", ":"))
    if st.session_state.include_csv_in_prompt:
        csv_text = slice_csv_for_prompt(df, st.session_state.csv_prompt_days) \
                   or st.session_state.csv_raw_text
        if csv_text:
            uprompt += "\n\n【補助データ: CSV（wide, 最新N日）】\n---\n" + csv_text + "\n---"
    return uprompt

def run_openai_chat(prompt: str, model_name: str):
    """
    戻り値: (parsed_json: dict, raw_text: str)
    """
    api_key = st.secrets["OPENAI_API_KEY"]
    base_url = st.secrets.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    messages = [
        {"role": "system", "content": "You are an expert ads budget allocation assistant. Output valid JSON only."},
        {"role": "user", "content": prompt},
    ]

    # 1) 公式SDK
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url=base_url)
        resp = client.chat.completions.create(
            model=model_name,
            response_format={"type": "json_object"},
            temperature=0.2,
            messages=messages,
        )
        raw = resp.choices[0].message.content
        return json.loads(raw), raw
    except ModuleNotFoundError:
        pass

    # 2) HTTP フォールバック
    body = {
        "model": model_name,
        "response_format": {"type": "json_object"},
        "temperature": 0.2,
        "messages": messages,
    }
    r = requests.post(
        f"{base_url}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=body, timeout=120
    )
    r.raise_for_status()
    data = r.json()
    raw = data["choices"][0]["message"]["content"]
    return json.loads(raw), raw

# 最終整合（share/amount/total）
def enforce_constraints(obj: dict) -> dict:
    medias = ["IGFB", "Google", "YT", "Tik"]
    total = max(0, int(obj.get("today_total_spend", 0)))
    alloc = obj.get("allocation", {}) or {}

    for m in medias:
        if m not in alloc:
            alloc[m] = {"share": 0, "amount": 0}
        alloc[m]["share"] = round(float(alloc[m].get("share", 0) or 0), 2)

    ssum = round(sum(alloc[m]["share"] for m in medias), 2)
    if ssum == 0 and total > 0:
        for m in medias:
            alloc[m]["share"] = 0.25
        ssum = 1.00
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
# UI: ヘッダ & サイドバー
# ---------------------------
st.title("AdAI 配分ビュー（手動取得 → 正規化 → プロンプト編集 → OpenAI A/B/C）")
jst_now = datetime.now(ZoneInfo("Asia/Tokyo"))
st.caption(f"JST 現在: {jst_now.strftime('%Y-%m-%d %H:%M')}")

with st.sidebar:
    st.markdown("### OpenAI 設定")
    default_model = st.secrets.get("OPENAI_MODEL", "gpt-4o-mini")
    model_name = st.text_input("モデル名", value=default_model, help="例: gpt-4o-mini / gpt-4o / gpt-4.1 / gpt-5 など権限のあるモデル")
    if "OPENAI_API_KEY" not in st.secrets:
        st.warning("Secrets に OPENAI_API_KEY を設定してください。")

    st.markdown("---")
    st.markdown("### プロンプトにCSVを含める")
    st.session_state.include_csv_in_prompt = st.checkbox("CSVを含める（最新N日）", value=st.session_state.include_csv_in_prompt)
    st.session_state.csv_prompt_days = st.slider("CSV: 最新N日", 1, 31, st.session_state.csv_prompt_days)

    st.markdown("---")
    st.markdown("### 予算/対象日のオーバーライド")
    if st.session_state.features is not None:
        f = st.session_state.features.get("facts", {})
        mb_default = int(f.get("month_budget") or 0)
        mb_in = st.number_input("月予算 (JPY)", min_value=0, step=10000, value=mb_default)
        st.session_state.month_budget_override = mb_in if mb_in != mb_default else None

        td_default = f.get("target_date") or jst_now.date().strftime("%Y-%m-%d")
        td_in = st.date_input("対象日（target_date）", value=pd.to_datetime(td_default).date())
        st.session_state.target_date_override = td_in.strftime("%Y-%m-%d")
    else:
        st.info("データ取得後に設定できます。")

with st.expander("手順", expanded=True):
    st.markdown(
        "1) **データ取得**: n8n から CSV または facts(JSON) を取得（配列で両方もOK）\n"
        "2) **正規化**: features（facts/channels）に整形。必要なら月予算/対象日を左サイドバーで上書き\n"
        "3) **A/B/Cプロンプト**を編集し、それぞれ推論実行 → 最終プロンプト/生出力/確定配分を比較"
    )

# ---------------------------
# データ取得
# ---------------------------
c1, c2, _ = st.columns([1, 1, 3])
with c1:
    if st.button("データ取得", type="primary"):
        with st.spinner("n8n から取得中…"):
            try:
                raw = fetch_latest_manual(False)
                st.session_state.raw = raw
                features, df = coerce_features_and_df(raw)
                st.session_state.df = df
                st.session_state.features = features
                # 初期プロンプトを投入（未設定時のみ）
                default_base = BASE_INSTRUCTIONS
                for key in ["A", "B", "C"]:
                    if st.session_state[f"prompt_{key}"] is None:
                        st.session_state[f"prompt_{key}"] = default_base
                # 結果リセット
                for key in ["A", "B", "C"]:
                    st.session_state[f"last_prompt_{key}"] = None
                    st.session_state[f"raw_output_{key}"] = None
                    st.session_state[f"result_{key}"] = None
                st.success("取得・正規化成功")
            except Exception as e:
                st.error(f"取得に失敗: {e}")

with c2:
    if st.button("再取得（強制）"):
        with st.spinner("キャッシュ無視で再取得…"):
            try:
                raw = fetch_latest_manual(True)
                st.session_state.raw = raw
                features, df = coerce_features_and_df(raw)
                st.session_state.df = df
                st.session_state.features = features
                for key in ["A", "B", "C"]:
                    st.session_state[f"last_prompt_{key}"] = None
                    st.session_state[f"raw_output_{key}"] = None
                    st.session_state[f"result_{key}"] = None
                st.success("再取得・正規化成功")
            except Exception as e:
                st.error(f"再取得に失敗: {e}")

# CSVプレビュー
if st.session_state.df is not None:
    st.subheader("CSV（wide）プレビュー")
    with st.expander("全期間（重い場合あり）", expanded=False):
        st.dataframe(st.session_state.df, use_container_width=True)
    cols = list(st.session_state.df.columns)
    k = min(7, len(cols))
    if k > 0:
        st.markdown("直近7日プレビュー")
        st.dataframe(st.session_state.df[cols[-k:]].head(12), use_container_width=True)

# features 表示 & オーバーライド適用
if st.session_state.features is not None:
    # オーバーライド同期
    facts = st.session_state.features.setdefault("facts", {})
    if st.session_state.month_budget_override is not None:
        facts["month_budget"] = int(st.session_state.month_budget_override)
    if st.session_state.target_date_override:
        facts["target_date"] = st.session_state.target_date_override

    st.subheader("features（LLM入力ベース）")
    st.code(json.dumps(st.session_state.features, ensure_ascii=False, indent=2))
else:
    st.info("まだデータ未取得です。**データ取得** を押してください。")

# ---------------------------
# 共通: 結果描画ヘルパ
# ---------------------------
def render_result(res: dict, label: str):
    if not res:
        st.info(f"{label}: まだ結果がありません。")
        return
    td = (res.get("report", {}) or {}).get("target_date") \
         or (st.session_state.features.get("facts", {}).get("target_date") if st.session_state.features else "")
    st.markdown(f"**{label} / 本日の配分（{td}）**")
    total = int(res.get("today_total_spend", 0) or 0)
    st.metric("総額", f"¥{total:,}")
    alloc = res.get("allocation", {}) or {}
    def g(m, k, default=0):
        return (alloc.get(m, {}) or {}).get(k, default)
    df_view = pd.DataFrame([
        {"media":"IGFB",  "share(%)": round((g("IGFB","share",0)*100),1),  "amount(¥)": int(g("IGFB","amount",0))},
        {"media":"Google","share(%)": round((g("Google","share",0)*100),1),"amount(¥)": int(g("Google","amount",0))},
        {"media":"YT",    "share(%)": round((g("YT","share",0)*100),1),    "amount(¥)": int(g("YT","amount",0))},
        {"media":"Tik",   "share(%)": round((g("Tik","share",0)*100),1),   "amount(¥)": int(g("Tik","amount",0))},
    ])
    st.dataframe(df_view, use_container_width=True)
    if res.get("reasoning_points"):
        st.markdown("**判断ポイント**")
        for p in res["reasoning_points"]:
            st.write("• " + str(p))
    if res.get("report"):
        st.markdown("**要約**")
        st.write(res["report"].get("executive_summary",""))

# ---------------------------
# Prompt Lab（A/B/C）
# ---------------------------
st.markdown("## Prompt Lab（A/B/C 比較）")
if "OPENAI_API_KEY" not in st.secrets:
    st.warning("Secrets に OPENAI_API_KEY を設定してください。")
elif st.session_state.features is None:
    st.info("先に **データ取得** を行ってください。")
else:
    cols = st.columns(3)
    for idx, key in enumerate(["A", "B", "C"]):
        with cols[idx]:
            st.markdown(f"### プロンプト {key}")
            st.session_state[f"prompt_{key}"] = st.text_area(
                f"ベースプロンプト {key}",
                value=st.session_state[f"prompt_{key}"] or BASE_INSTRUCTIONS,
                height=280,
                key=f"ta_{key}"
            )
            run = st.button(f"{key} で推論を実行", key=f"run_{key}", use_container_width=True)
            if run:
                try:
                    # 最終Userプロンプトを組み立て（CSV同梱オプション適用）
                    user_prompt = compose_user_prompt(st.session_state[f"prompt_{key}"], st.session_state.features, st.session_state.df)
                    st.session_state[f"last_prompt_{key}"] = user_prompt

                    with st.spinner(f"OpenAI {model_name} で推論中…"):
                        result_obj, raw_text = run_openai_chat(user_prompt, model_name=model_name)
                        result_obj = enforce_constraints(result_obj)
                        st.session_state[f"raw_output_{key}"] = raw_text
                        st.session_state[f"result_{key}"] = result_obj
                        st.success(f"{key}: 推論成功")
                except requests.exceptions.HTTPError as e:
                    st.error(f"{key}: OpenAI API エラー: {e}")
                except Exception as e:
                    st.error(f"{key}: 推論に失敗: {e}")

            # プロンプト/生出力の可視化
            if st.session_state[f"last_prompt_{key}"]:
                with st.expander(f"最終プロンプト {key}（CSV同梱後）", expanded=False):
                    st.code(st.session_state[f"last_prompt_{key}"])
            if st.session_state[f"raw_output_{key}"]:
                with st.expander(f"LLM 生出力 {key}（加工前）", expanded=False):
                    st.code(st.session_state[f"raw_output_{key}"])

            # 結果描画
            render_result(st.session_state[f"result_{key}"], f"結果 {key}")

st.caption("※ CSV（wide）または facts(JSON) を取り込み、必要に応じて月予算/対象日を上書き。CSVは最新N日だけプロンプトへ同梱可能。A/B/Cのプロンプト差分を比較できます。")
