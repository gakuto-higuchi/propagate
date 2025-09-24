# app.py
import io, json, time
import requests, pandas as pd, streamlit as st
from openai import OpenAI  # 公式SDK

st.set_page_config(page_title="AdAI 配分ビュー（CSV/JSON→要約→OpenAI SDK）", layout="wide")

# =========================
# セッション状態
# =========================
if "raw" not in st.session_state:      st.session_state.raw = None
if "df" not in st.session_state:       st.session_state.df = None
if "features" not in st.session_state: st.session_state.features = None
if "result" not in st.session_state:   st.session_state.result = None

# ---- Prompt/Model UI の既定値 ----
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
DEFAULT_SYS = "You are an expert ads budget allocation assistant. Output valid JSON only."

if "sys_prompt" not in st.session_state:  st.session_state.sys_prompt = DEFAULT_SYS
if "base_prompt" not in st.session_state: st.session_state.base_prompt = BASE_INSTRUCTIONS
if "model" not in st.session_state:       st.session_state.model = st.secrets.get("OPENAI_MODEL", "gpt-4o-mini")
if "temperature" not in st.session_state: st.session_state.temperature = float(st.secrets.get("OPENAI_TEMPERATURE", 0.2))

# =========================
# n8n取得（手動のみ）
# =========================
def _http_get_latest(timeout_s: int = 20):
    url = st.secrets["N8N_JSON_URL"]
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

# =========================
# 入力正規化（CSV or facts型）
# =========================
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

    cost_all = df.loc["コスト_ALL"].dropna() if "コスト_ALL" in df.index else pd.Series(dtype=float)

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

    facts = dict(obj["facts"])
    channels = {}
    if "channels" in facts and isinstance(facts["channels"], dict):
        channels = facts["channels"]
        facts = {k: v for k, v in facts.items() if k != "channels"}
    elif "channels" in obj and isinstance(obj["channels"], dict):
        channels = obj["channels"]

    return {"facts": facts, "channels": channels}

def coerce_features_and_df(raw):
    """n8nの返却を判定して features / df を返す"""
    df = None
    features = None

    # Case A: CSV型
    if isinstance(raw, dict) and isinstance(raw.get("csv"), dict) and "wide" in raw["csv"]:
        csv_text = raw["csv"]["wide"]
        meta = raw.get("meta", {}) or {}
        if not csv_text:
            raise ValueError("csv.wide が空です")
        df = parse_wide_csv(csv_text)
        features = build_features_from_csv(df, meta)
        return features, df

    # Case B: facts型（配列 or 単体）
    try:
        features = build_features_from_factsish(raw)
        return features, df
    except Exception:
        pass

    raise ValueError("未対応のペイロード形式です")

# =========================
# OpenAI SDK 呼び出し
# =========================
# Chat Completions + JSONモード（公式仕様）
# ref: response_format={"type":"json_object"} / messages / model 指定
# https://platform.openai.com/docs/api-reference/chat/create
_client = None
def _get_openai_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = st.secrets["OPENAI_API_KEY"].strip()
        base_url = st.secrets.get("OPENAI_BASE_URL", "https://api.openai.com/v1").strip()
        _client = OpenAI(api_key=api_key, base_url=base_url)
    return _client

def build_llm_input(features: dict) -> str:
    base = st.session_state.get("base_prompt", BASE_INSTRUCTIONS)
    return base + json.dumps(features, ensure_ascii=False, separators=(",", ":"))

def run_openai_chat(prompt: str) -> dict:
    client = _get_openai_client()
    model = st.session_state.get("model") or "gpt-4o-mini"
    temp = float(st.session_state.get("temperature", 0.2))
    sys_prompt = st.session_state.get("sys_prompt", DEFAULT_SYS)

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0.2 if temp is None else temp,
    )
    return json.loads(resp.choices[0].message.content)

# =========================
# 整合処理（share/amount/total）
# =========================
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

# =========================
# UI
# =========================
st.title("AdAI 配分ビュー（手動取得 → 要約/統合 → OpenAI SDK）")

with st.expander("手順", expanded=True):
    st.markdown(
        "1) **データ取得**: n8n から CSV または JSON（facts型）を取得\n"
        "2) **統合**: CSVなら要約を生成、facts型ならそのまま正規化（features）\n"
        "3) **推論**: OpenAI SDK（Chat Completions / JSONモード）で配分JSONを生成"
    )

# ---- プロンプト & モデル編集UI ----
with st.expander("プロンプト & モデル設定（保存してから推論）", expanded=False):
    with st.form("prompt_form"):
        sys_prompt = st.text_area("System prompt", value=st.session_state.sys_prompt, height=80)
        base_prompt = st.text_area("User prompt（featuresは末尾に自動結合）",
                                   value=st.session_state.base_prompt, height=280)
        cols = st.columns([1,1,1,2])
        model = cols[0].text_input("Model", value=st.session_state.model, placeholder="gpt-4o-mini")
        temperature = cols[1].slider("Temperature", 0.0, 1.0, float(st.session_state.temperature), 0.05)
        save = cols[2].form_submit_button("保存")
        reset = cols[3].form_submit_button("デフォルトに戻す")
        if save:
            st.session_state.sys_prompt = sys_prompt
            st.session_state.base_prompt = base_prompt
            st.session_state.model = model
            st.session_state.temperature = float(temperature)
            st.success("設定を保存しました")
        if reset:
            st.session_state.sys_prompt = DEFAULT_SYS
            st.session_state.base_prompt = BASE_INSTRUCTIONS
            st.session_state.model = "gpt-4o-mini"
            st.session_state.temperature = 0.2
            st.info("デフォルトに戻しました")

    # オプション：設定のエクスポート/インポート
    cfg = {
        "sys_prompt": st.session_state.sys_prompt,
        "base_prompt": st.session_state.base_prompt,
        "model": st.session_state.model,
        "temperature": st.session_state.temperature,
    }
    st.download_button("設定をダウンロード",
                       data=json.dumps(cfg, ensure_ascii=False, indent=2),
                       file_name="prompt_config.json", mime="application/json")
    up = st.file_uploader("設定を読み込む（JSON）", type=["json"])
    if up is not None:
        try:
            cfg2 = json.load(up)
            st.session_state.sys_prompt = cfg2.get("sys_prompt", st.session_state.sys_prompt)
            st.session_state.base_prompt = cfg2.get("base_prompt", st.session_state.base_prompt)
            st.session_state.model = cfg2.get("model", st.session_state.model)
            st.session_state.temperature = float(cfg2.get("temperature", st.session_state.temperature))
            st.success("設定を読み込みました")
        except Exception as e:
            st.warning(f"設定の読み込みに失敗: {e}")

# ---- 取得ボタン ----
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
                st.session_state.result = None
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
                st.session_state.result = None
                st.success("再取得・正規化成功")
            except Exception as e:
                st.error(f"再取得に失敗: {e}")

# ---- プレビュー ----
if st.session_state.df is not None:
    st.subheader("CSVプレビュー（先頭数行 × 直近7日）")
    cols = list(st.session_state.df.columns)
    k = min(7, len(cols))
    preview_cols = cols[-k:] if k > 0 else []
    st.dataframe(st.session_state.df[preview_cols].head(10), use_container_width=True)

if st.session_state.features is not None:
    st.subheader("features（LLM入力）")
    st.code(json.dumps(st.session_state.features, ensure_ascii=False, indent=2))
else:
    st.info("まだデータ未取得です。**データ取得** を押してください。")

# ---- 推論 ----
st.markdown("### 推論（OpenAI SDK: Chat Completions / JSONモード）")
if "OPENAI_API_KEY" not in st.secrets:
    st.warning("Secrets に OPENAI_API_KEY を設定してください。")
else:
    disabled = st.session_state.features is None
    if st.button("推論を実行", disabled=disabled):
        if disabled:
            st.warning("先にデータ取得を行ってください。")
        else:
            with st.spinner("モデル推論中…"):
                try:
                    prompt = build_llm_input(st.session_state.features)
                    result = run_openai_chat(prompt)
                    result = enforce_constraints(result)
                    st.session_state.result = result
                    st.success("推論成功")
                except requests.exceptions.HTTPError as e:
                    st.error(f"OpenAI API エラー: {e}")
                except Exception as e:
                    st.error(f"推論に失敗: {e}")

# ---- 結果表示 ----
res = st.session_state.result
if res:
    td = res.get("report", {}).get("target_date") \
         or (st.session_state.features.get("facts", {}).get("target_date") if st.session_state.features else "")
    st.subheader(f"本日の配分（{td}）")
    st.metric("総額", f"¥{int(res.get('today_total_spend', 0)):,}")

    alloc = res.get("allocation", {})
    df_view = pd.DataFrame([
        {"media": "IGFB",   "share(%)": round(alloc.get("IGFB",{}).get("share",0)*100, 1),   "amount(¥)": alloc.get("IGFB",{}).get("amount",0)},
        {"media": "Google", "share(%)": round(alloc.get("Google",{}).get("share",0)*100, 1), "amount(¥)": alloc.get("Google",{}).get("amount",0)},
        {"media": "YT",     "share(%)": round(alloc.get("YT",{}).get("share",0)*100, 1),     "amount(¥)": alloc.get("YT",{}).get("amount",0)},
        {"media": "Tik",    "share(%)": round(alloc.get("Tik",{}).get("share",0)*100, 1),    "amount(¥)": alloc.get("Tik",{}).get("amount",0)},
    ])
    st.dataframe(df_view, use_container_width=True)

    if res.get("reasoning_points"):
        st.markdown("#### 判断ポイント")
        for p in res["reasoning_points"]:
            st.write("• " + str(p))

    if res.get("report"):
        st.markdown("#### 要約")
        st.write(res["report"].get("executive_summary", ""))

    st.download_button(
        "結果JSONをダウンロード",
        data=json.dumps(res, ensure_ascii=False, indent=2),
        file_name="allocation_result.json",
        mime="application/json"
    )

st.caption("※ CSV型／facts型どちらのペイロードにも対応。初期表示では外部アクセスなし。")
