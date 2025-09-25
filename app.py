# app.py
# AdAI 配分ビュー（facts + CSV 必須 / プロンプトA/B/C比較）
# - 初期表示で外部アクセスなし
# - n8n の返却は「facts と csv.wide の両方」が必須
# - BASE_INSTRUCTIONS（共通ベース指示）を含む全プロンプトを UI で編集可
# - A/B/C で model / temperature 等の任意パラメータは「指定する」ON時のみ送信
# - 3カラム/タブ切替、最終投入プロンプト&生出力&整合後JSON 可視化

import io, json, time, math
from datetime import datetime, timezone, timedelta, date
import requests, pandas as pd, streamlit as st
# ---------- 表示ヘルパー（理由・レポートをまとめて描画） ----------
def _fmt_num(x):
    if x is None: return "—"
    if isinstance(x, (int,)) and not isinstance(x, bool): return f"{x:,}"
    try:
        xf = float(x)
        # 小数は状況に応じて丸め（CPA/CPC/CVRなどの見栄え配慮）
        if abs(xf) >= 100: return f"{xf:,.0f}"
        return f"{xf:,.3f}".rstrip("0").rstrip(".")
    except Exception:
        return str(x)

def render_reasoning_and_report(res: dict, sid: str):
    # 1) reasoning_points
    if res.get("reasoning_points"):
        st.markdown(f"**[{sid}] 判断ポイント**")
        for p in res["reasoning_points"]:
            st.write("• " + str(p))

    rpt = res.get("report") or {}
    if not rpt:
        return

    # 2) Executive summary
    if rpt.get("executive_summary"):
        st.markdown(f"**[{sid}] 要約**")
        st.write(rpt["executive_summary"])

    # 3) Pacing decision
    pdct = rpt.get("pacing_decision") or {}
    if pdct:
        c1, c2 = st.columns([1,3])
        with c1:
            st.metric(f"[{sid}] chosen_today_total", _fmt_num(pdct.get("chosen_today_total", 0)))
        with c2:
            st.write(f"**Why**: {pdct.get('why','')}")

    # 4) Channel signals（表に整形）
    chs = rpt.get("channel_signals") or {}
    if chs:
        rows = []
        for media in ["IGFB","Google","YT","Tik"]:
            d = chs.get(media, {})
            last7 = d.get("last7", {}) or {}
            rows.append({
                "media": media,
                "stance": d.get("stance",""),
                "confidence": d.get("confidence",""),
                "median_CPA": _fmt_num(last7.get("median_CPA")),
                "median_CVR": _fmt_num(last7.get("median_CVR")),
                "median_CPC": _fmt_num(last7.get("median_CPC")),
                "clicks_sum": _fmt_num(last7.get("clicks_sum")),
                "cv_sum":     _fmt_num(last7.get("cv_sum")),
            })
        st.markdown(f"**[{sid}] 媒体シグナル（last7）**")
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

        # 媒体ごとの補足（why）があれば下に列挙
        details = []
        for media in ["IGFB","Google","YT","Tik"]:
            why = (chs.get(media) or {}).get("why")
            if why: details.append(f"- **{media}**: {why}")
        if details:
            st.markdown("\n".join(details))

    # 5) Decision flow
    dflow = rpt.get("decision_flow") or []
    if dflow:
        st.markdown(f"**[{sid}] 意思決定フロー**")
        for i, step in enumerate(dflow, 1):
            data_used = ", ".join(step.get("data_used", []))
            logic = step.get("logic","")
            st.write(f"{i}. **{step.get('step','')}** — *data_used:* {data_used} / *logic:* {logic}")

    # 6) Final allocation check（期待CV）
    fac = rpt.get("final_allocation_check") or {}
    exp = fac.get("expected_cv") or {}
    if exp:
        st.markdown(f"**[{sid}] 期待CV（合算チェック）**")
        df_ecv = pd.DataFrame([{
            "IGFB": _fmt_num(exp.get("IGFB")),
            "Google": _fmt_num(exp.get("Google")),
            "YT": _fmt_num(exp.get("YT")),
            "Tik": _fmt_num(exp.get("Tik")),
        }])
        st.dataframe(df_ecv, use_container_width=True)
        if fac.get("notes"):
            st.caption(f"Notes: {fac['notes']}")

    # 7) Risks & Actions
    ra = rpt.get("risks_and_actions") or {}
    risks = ra.get("risks") or []
    acts  = ra.get("next_actions") or []
    if risks or acts:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**[{sid}] リスク**")
            for r in risks:
                st.write("• " + str(r))
        with c2:
            st.markdown(f"**[{sid}] 次アクション**")
            for a in acts:
                st.write("• " + str(a))


# -------------------------------------------------
# 基本設定 & 軽い CSS（タイトルの食い込み/レイアウト崩れ対策）
# -------------------------------------------------
st.set_page_config(page_title="AdAI 配分ビュー（facts+CSV 必須 / A-B-C 比較）", layout="wide")
st.markdown("""
<style>
/* タイトルが切れないように上マージンを確保 */
.block-container { padding-top: 1.5rem; max-width: 1400px; }
/* 見出しの余白を安定化 */
h1, h2, h3 { margin-top: .4rem; margin-bottom: .6rem; }
/* dataframes は幅に追従しつつ最低高さを保持 */
div[data-testid="stDataFrame"] { min-height: 340px; }
</style>
""", unsafe_allow_html=True)
JST = timezone(timedelta(hours=9))

# -------------------------------------------------
# セッション状態
# -------------------------------------------------
if "raw" not in st.session_state:        st.session_state.raw = None
if "facts" not in st.session_state:      st.session_state.facts = None
if "channels" not in st.session_state:   st.session_state.channels = {}
if "csv_text" not in st.session_state:   st.session_state.csv_text = None
if "meta" not in st.session_state:       st.session_state.meta = {}
if "df" not in st.session_state:         st.session_state.df = None
if "slots" not in st.session_state:      st.session_state.slots = {}
if "base_instructions" not in st.session_state:
    st.session_state.base_instructions = """あなたは広告予算配分の最適化アシスタントです。出力は厳密なJSONオブジェクトのみで返してください（余分な文章・コードフェンス不可）。

【目的】
- 入力 facts/channels と補助のCSV断片（任意）を踏まえ、CPA最適化の観点で「本日の総投資額」と「媒体別配分」を決定する。
- 月予算や candidates（yesterday_total/avg_last3/median_last7 等）があれば参照（抑制も可）。

【前提・制約】
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

SLOT_IDS = ["A", "B", "C"]

# -------------------------------------------------
# n8n 取得（手動）
# -------------------------------------------------
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
    last_err = None
    for i, t in enumerate([10, 20, 40], start=1):
        try:
            return _http_get_latest(timeout_s=t)
        except requests.exceptions.ReadTimeout as e:
            last_err = e
            time.sleep(0.7 * i)
    raise last_err or requests.exceptions.ReadTimeout("n8n webhook timeout")

# -------------------------------------------------
# 受信 → facts/channels + csv を必須抽出
# -------------------------------------------------
def parse_wide_csv(csv_text: str) -> pd.DataFrame:
    df = pd.read_csv(io.StringIO(csv_text))
    if "variable" not in df.columns:
        raise ValueError("CSVに 'variable' 列がありません")
    df = df.set_index("variable")
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

def coerce_require_facts_and_csv(raw):
    facts, csv_text, meta, channels = None, None, {}, {}
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict):
                if facts is None and "facts" in item:
                    facts = item["facts"]
                    if "channels" in facts and isinstance(facts["channels"], dict):
                        channels = facts["channels"]
                        facts = {k: v for k, v in facts.items() if k != "channels"}
                if csv_text is None and isinstance(item.get("csv"), dict) and "wide" in item["csv"]:
                    csv_text = item["csv"]["wide"]
                    meta = item.get("meta", {}) or meta
    elif isinstance(raw, dict):
        if "facts" in raw:
            facts = raw["facts"]
            if "channels" in facts and isinstance(facts["channels"], dict):
                channels = facts["channels"]
                facts = {k: v for k, v in facts.items() if k != "channels"}
        if isinstance(raw.get("csv"), dict) and "wide" in raw["csv"]:
            csv_text = raw["csv"]["wide"]
            meta = raw.get("meta", {}) or meta
    else:
        raise ValueError("未対応のペイロード形式です")

    if not isinstance(facts, dict):
        raise ValueError("facts が見つかりません（必須）")
    if not csv_text or not isinstance(csv_text, str):
        raise ValueError("csv.wide が見つかりません（必須）")

    return facts, channels, csv_text, meta

# -------------------------------------------------
# CSVユーティリティ
# -------------------------------------------------
def df_latest_days(df: pd.DataFrame, n_days: int) -> pd.DataFrame:
    cols = [c for c in df.columns if str(c)[:4].isdigit()]
    use_cols = cols[-n_days:] if n_days > 0 else cols
    return df[use_cols]

def df_to_csv_text(df_slice: pd.DataFrame) -> str:
    out = df_slice.copy()
    out.insert(0, "variable", out.index)
    return out.to_csv(index=False)

# -------------------------------------------------
# プロンプト生成 & OpenAI 呼び出し
# -------------------------------------------------
def build_features_for_prompt(facts: dict, channels: dict, override_budget, override_target):
    f = dict(facts)
    if override_budget is not None:
        f["month_budget"] = float(override_budget)
    if override_target is not None:
        f["target_date"] = override_target.strftime("%Y-%m-%d")
    return {"facts": f, "channels": channels or {}}

def make_final_prompt_text(system_text: str, user_text: str, features_obj: dict, include_csv: bool, csv_payload: str|None):
    base_instr = st.session_state.base_instructions or ""
    body = base_instr + "\n\n" + json.dumps(features_obj, ensure_ascii=False, separators=(",", ":"))
    if include_csv and csv_payload:
        body += "\n\n【CSV】\n" + csv_payload
    if user_text and user_text.strip():
        body += "\n\n【追加指示】\n" + user_text.strip()
    # UIで見えるように合成版も保持
    visible = f"<<SYSTEM>>\n{system_text.strip()}\n\n<<USER>>\n{body}"
    return body, visible

def run_openai_chat(slot_cfg: dict, user_prompt: str) -> str:
    api_key  = st.secrets["OPENAI_API_KEY"]
    base_url = st.secrets.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    model    = slot_cfg["model"]
    system_text = slot_cfg["system"]

    messages = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": user_prompt},
    ]

    # 任意パラメータは「指定する」ON時のみ kwargs/body に入れる
    def maybe_put(d, key, flag, value, cast=float):
        if flag and value is not None:
            d[key] = cast(value)

    # 1) SDK
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url=base_url)
        kwargs = {"model": model, "response_format": {"type": "json_object"}, "messages": messages}
        maybe_put(kwargs, "temperature",        slot_cfg.get("use_temperature"),        slot_cfg.get("temperature"),        float)
        maybe_put(kwargs, "top_p",              slot_cfg.get("use_top_p"),              slot_cfg.get("top_p"),              float)
        maybe_put(kwargs, "max_tokens",         slot_cfg.get("use_max_tokens"),         slot_cfg.get("max_tokens"),         int)
        maybe_put(kwargs, "seed",               slot_cfg.get("use_seed"),               slot_cfg.get("seed"),               int)
        maybe_put(kwargs, "presence_penalty",   slot_cfg.get("use_presence_penalty"),   slot_cfg.get("presence_penalty"),   float)
        maybe_put(kwargs, "frequency_penalty",  slot_cfg.get("use_frequency_penalty"),  slot_cfg.get("frequency_penalty"),  float)

        resp = client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content
    except ModuleNotFoundError:
        pass

    # 2) HTTP フォールバック
    body = {"model": model, "response_format": {"type": "json_object"}, "messages": messages}
    maybe_put(body, "temperature",        slot_cfg.get("use_temperature"),        slot_cfg.get("temperature"),        float)
    maybe_put(body, "top_p",              slot_cfg.get("use_top_p"),              slot_cfg.get("top_p"),              float)
    maybe_put(body, "max_tokens",         slot_cfg.get("use_max_tokens"),         slot_cfg.get("max_tokens"),         int)
    maybe_put(body, "seed",               slot_cfg.get("use_seed"),               slot_cfg.get("seed"),               int)
    maybe_put(body, "presence_penalty",   slot_cfg.get("use_presence_penalty"),   slot_cfg.get("presence_penalty"),   float)
    maybe_put(body, "frequency_penalty",  slot_cfg.get("use_frequency_penalty"),  slot_cfg.get("frequency_penalty"),  float)

    r = requests.post(f"{base_url}/chat/completions",
                      headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                      json=body, timeout=120)
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"]

def enforce_constraints(obj: dict) -> dict:
    medias = ["IGFB","Google","YT","Tik"]
    total = max(0, int(obj.get("today_total_spend", 0)))
    alloc = obj.get("allocation", {}) or {}
    for m in medias:
        if m not in alloc:
            alloc[m] = {"share": 0, "amount": 0}
        try:
            alloc[m]["share"] = round(float(alloc[m].get("share", 0) or 0), 2)
        except Exception:
            alloc[m]["share"] = 0.0
    ssum = round(sum(alloc[m]["share"] for m in medias), 2)
    if ssum == 0 and total > 0:
        for m in medias: alloc[m]["share"] = 0.25
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

# -------------------------------------------------
# ヘッダ & サイドバー
# -------------------------------------------------
st.title("AdAI 配分ビュー（facts+CSV 必須 / A-B-C プロンプト比較）")
st.caption(f"現在時刻（JST）: {datetime.now(JST).strftime('%Y-%m-%d %H:%M:%S')}")

st.sidebar.header("共通設定")
if "OPENAI_API_KEY" not in st.secrets:
    st.sidebar.error("Secrets に OPENAI_API_KEY を設定してください。")

# facts の上書き
use_override_budget = st.sidebar.checkbox("月予算の上書きを有効化", value=False)
override_budget = st.sidebar.number_input("月予算（上書き値 / 任意）", min_value=0.0, step=1000.0, format="%.0f", value=0.0)
use_override_target = st.sidebar.checkbox("対象日の上書きを有効化", value=False)
override_target_date = st.sidebar.date_input("対象日（上書き値 / 任意）", value=None, min_value=date(2000,1,1), max_value=date(2100,12,31), format="YYYY-MM-DD")

st.sidebar.divider()
st.sidebar.subheader("BASE_INSTRUCTIONS（共通ベース指示）")
st.session_state.base_instructions = st.sidebar.text_area(
    "全スロット共通で 'user' に渡すベース指示（編集可）",
    value=st.session_state.base_instructions,
    height=260
)
if st.sidebar.button("BASE_INSTRUCTIONS を既定に戻す"):
    st.session_state.base_instructions = st.session_state.base_instructions.split("\n【入力（features: facts/channels）】")[0] + "\n【入力（features: facts/channels）】"

st.sidebar.divider()
st.sidebar.write("**CSV同梱の既定（各スロットでも変更可）**")
csv_default_include = st.sidebar.checkbox("CSVをプロンプトへ含める（既定）", value=True)
csv_default_mode = st.sidebar.radio("CSV同梱モード（既定）", ["全期間", "最新N日"], horizontal=True, index=0)
csv_default_days = st.sidebar.slider("最新N日を選ぶ（既定）", 3, 31, 7, 1)

# -------------------------------------------------
# 取得 / 再取得
# -------------------------------------------------
c1, c2, c3 = st.columns([1,1,2])
with c1:
    if st.button("データ取得", type="primary"):
        with st.spinner("n8n から取得中…"):
            try:
                raw = fetch_latest_manual(False)
                facts, channels, csv_text, meta = coerce_require_facts_and_csv(raw)
                df = parse_wide_csv(csv_text)
                st.session_state.raw = raw
                st.session_state.facts = facts
                st.session_state.channels = channels
                st.session_state.csv_text = csv_text
                st.session_state.meta = meta
                st.session_state.df = df
                st.success("取得成功：facts / channels / CSV を読み込みました。")
            except Exception as e:
                st.error(f"取得に失敗: {e}")
with c2:
    if st.button("再取得（強制）"):
        with st.spinner("キャッシュ無視で再取得…"):
            try:
                raw = fetch_latest_manual(True)
                facts, channels, csv_text, meta = coerce_require_facts_and_csv(raw)
                df = parse_wide_csv(csv_text)
                st.session_state.raw = raw
                st.session_state.facts = facts
                st.session_state.channels = channels
                st.session_state.csv_text = csv_text
                st.session_state.meta = meta
                st.session_state.df = df
                st.success("再取得成功")
            except Exception as e:
                st.error(f"再取得に失敗: {e}")
with c3:
    st.info("注意：facts と CSV の両方が無いと推論できません。")

# -------------------------------------------------
# プレビュー
# -------------------------------------------------
if st.session_state.df is not None:
    st.subheader("CSVプレビュー（全期間・全行）")
    st.dataframe(st.session_state.df, use_container_width=True, height=560)

if st.session_state.facts is not None:
    st.subheader("facts（元データ）")
    st.code(json.dumps(st.session_state.facts, ensure_ascii=False, indent=2))

# -------------------------------------------------
# スロット初期化（A/B/C）
# -------------------------------------------------
def ensure_slots():
    defaults = {
        "model": st.secrets.get("OPENAI_MODEL", "gpt-4o-mini"),
        # 任意パラメータ（指定する=OFF → 送らない）
        "use_temperature": False, "temperature": 1.0,
        "use_top_p": False, "top_p": 1.0,
        "use_max_tokens": False, "max_tokens": 0,
        "use_seed": False, "seed": 0,
        "use_presence_penalty": False, "presence_penalty": 0.0,
        "use_frequency_penalty": False, "frequency_penalty": 0.0,
        # プロンプト
        "system": "You are an expert ads budget allocation assistant. Output valid JSON only.",
        "user": "数値根拠を明示しつつ、現実的な配分を提案してください。",
        # CSV 同梱設定
        "include_csv": csv_default_include,
        "csv_mode": csv_default_mode,  # 全期間 / 最新N日
        "csv_days": csv_default_days,
        # 実行結果
        "final_prompt_preview": None,
        "raw_output": None,
        "result": None,
    }
    for sid in SLOT_IDS:
        if sid not in st.session_state.slots:
            st.session_state.slots[sid] = dict(defaults)
ensure_slots()

# -------------------------------------------------
# スロット編集 UI
# -------------------------------------------------
st.markdown("## Prompt Lab（A/B/C）— モデル/温度/プロンプト等を完全手動で設定")

def slot_editor(sid: str):
    cfg = st.session_state.slots[sid]
    with st.expander(f"スロット {sid} の設定・実行", expanded=(sid=="A")):
        c1, c2 = st.columns([1.4, 1])
        with c1:
            cfg["model"] = st.text_input(f"[{sid}] モデル名", value=cfg["model"], key=f"{sid}_model")
        with c2:
            cfg["include_csv"] = st.checkbox(f"[{sid}] CSVをプロンプトへ含める", value=cfg["include_csv"], key=f"{sid}_incl_csv")

        if cfg["include_csv"]:
            c3, c4 = st.columns([1,1.5])
            with c3:
                cfg["csv_mode"] = st.radio(f"[{sid}] CSV同梱モード", ["全期間", "最新N日"], index=(0 if cfg["csv_mode"]=="全期間" else 1), horizontal=True, key=f"{sid}_csv_mode")
            with c4:
                if cfg["csv_mode"] == "最新N日":
                    cfg["csv_days"] = st.slider(f"[{sid}] 最新N日", 3, 31, cfg["csv_days"], 1, key=f"{sid}_csv_days")

        cfg["system"] = st.text_area(f"[{sid}] System プロンプト", value=cfg["system"], height=100, key=f"{sid}_sys")
        cfg["user"]   = st.text_area(f"[{sid}] User プロンプト（追加指示）", value=cfg["user"], height=140, key=f"{sid}_usr")

        with st.expander(f"[{sid}] 高度なパラメータ（任意 / 指定時のみ送信）", expanded=False):
            cA, cB, cC = st.columns(3)
            with cA:
                cfg["use_temperature"] = st.checkbox(f"[{sid}] temperature を指定", value=cfg["use_temperature"], key=f"{sid}_use_temp")
                if cfg["use_temperature"]:
                    cfg["temperature"] = st.number_input(f"[{sid}] temperature", min_value=0.0, max_value=2.0, step=0.1, value=float(cfg["temperature"]), key=f"{sid}_temp_val")
            with cB:
                cfg["use_top_p"] = st.checkbox(f"[{sid}] top_p を指定", value=cfg["use_top_p"], key=f"{sid}_use_topp")
                if cfg["use_top_p"]:
                    cfg["top_p"] = st.number_input(f"[{sid}] top_p", min_value=0.0, max_value=1.0, step=0.05, value=float(cfg["top_p"]), key=f"{sid}_topp_val")
            with cC:
                cfg["use_max_tokens"] = st.checkbox(f"[{sid}] max_tokens を指定", value=cfg["use_max_tokens"], key=f"{sid}_use_maxtok")
                if cfg["use_max_tokens"]:
                    cfg["max_tokens"] = st.number_input(f"[{sid}] max_tokens", min_value=1, step=50, value=int(cfg["max_tokens"] or 256), key=f"{sid}_maxtok_val")

            cD, cE, cF = st.columns(3)
            with cD:
                cfg["use_seed"] = st.checkbox(f"[{sid}] seed を指定", value=cfg["use_seed"], key=f"{sid}_use_seed")
                if cfg["use_seed"]:
                    cfg["seed"] = st.number_input(f"[{sid}] seed", min_value=0, step=1, value=int(cfg["seed"]), key=f"{sid}_seed_val")
            with cE:
                cfg["use_presence_penalty"] = st.checkbox(f"[{sid}] presence_penalty を指定", value=cfg["use_presence_penalty"], key=f"{sid}_use_pp")
                if cfg["use_presence_penalty"]:
                    cfg["presence_penalty"] = st.number_input(f"[{sid}] presence_penalty", min_value=-2.0, max_value=2.0, step=0.1, value=float(cfg["presence_penalty"]), key=f"{sid}_pp_val")
            with cF:
                cfg["use_frequency_penalty"] = st.checkbox(f"[{sid}] frequency_penalty を指定", value=cfg["use_frequency_penalty"], key=f"{sid}_use_fp")
                if cfg["use_frequency_penalty"]:
                    cfg["frequency_penalty"] = st.number_input(f"[{sid}] frequency_penalty", min_value=-2.0, max_value=2.0, step=0.1, value=float(cfg["frequency_penalty"]), key=f"{sid}_fp_val")

        # 実行
        run_btn = st.button(f"[{sid}] 推論を実行", key=f"{sid}_run", use_container_width=True)
        if run_btn:
            if st.session_state.facts is None or st.session_state.df is None:
                st.error("facts と CSV の両方が必要です。先に『データ取得』を行ってください。")
            elif "OPENAI_API_KEY" not in st.secrets:
                st.error("Secrets に OPENAI_API_KEY を設定してください。")
            else:
                with st.spinner(f"{sid}: OpenAI で推論中…"):
                    try:
                        ov_budget = float(override_budget) if use_override_budget else None
                        ov_target = override_target_date if use_override_target else None
                        features = build_features_for_prompt(st.session_state.facts, st.session_state.channels, ov_budget, ov_target)

                        # CSV payload
                        csv_payload = None
                        if cfg["include_csv"]:
                            if cfg["csv_mode"] == "全期間":
                                csv_payload = df_to_csv_text(st.session_state.df)
                            else:
                                csv_payload = df_to_csv_text(df_latest_days(st.session_state.df, int(cfg["csv_days"])))

                        user_prompt, visible_prompt = make_final_prompt_text(
                            cfg["system"], cfg["user"], features, cfg["include_csv"], csv_payload
                        )
                        cfg["final_prompt_preview"] = visible_prompt

                        raw_text = run_openai_chat(cfg, user_prompt)
                        cfg["raw_output"] = raw_text

                        # JSON 整形
                        try:
                            obj = json.loads(raw_text.strip().strip("`"))
                        except json.JSONDecodeError:
                            s = raw_text.strip()
                            if s.startswith("```"):
                                parts = s.split("```")
                                if len(parts) >= 3:
                                    s = parts[1]
                                s = s.replace("json", "", 1).strip()
                            obj = json.loads(s.strip("` \n\r\t"))

                        cfg["result"] = enforce_constraints(obj)
                        st.success(f"{sid}: 推論成功")

                    except requests.exceptions.HTTPError as e:
                        st.error(f"{sid}: OpenAI API エラー: {e}")
                    except json.JSONDecodeError as e:
                        st.error(f"{sid}: LLM出力がJSONとして解釈できませんでした: {e}")
                        st.code(cfg.get("raw_output", "")[:1200])
                    except Exception as e:
                        st.error(f"{sid}: 推論に失敗: {e}")

        # プレビュー（最終プロンプト / 生出力 / 整合後）
        if cfg.get("final_prompt_preview"):
            st.markdown(f"**[{sid}] 最終プロンプト（実投入内容の可視化）**")
            st.code(cfg["final_prompt_preview"])
        if cfg.get("raw_output") is not None:
            st.markdown(f"**[{sid}] LLM 生出力（加工前）**")
            st.code(cfg["raw_output"])
        if cfg.get("result"):
            res = cfg["result"]
            st.markdown(f"**[{sid}] 整合後の配分結果**")
            td = res.get("report", {}).get("target_date") or st.session_state.facts.get("target_date", "")
            st.metric(f"[{sid}] 総額", f"¥{int(res.get('today_total_spend', 0)):,}")

            alloc = res.get("allocation", {})
            df_view = pd.DataFrame([
                {"media":"IGFB",  "share(%)": round((alloc.get("IGFB",{}).get("share",0))*100,1),  "amount(¥)": alloc.get("IGFB",{}).get("amount",0)},
                {"media":"Google","share(%)": round((alloc.get("Google",{}).get("share",0))*100,1),"amount(¥)": alloc.get("Google",{}).get("amount",0)},
                {"media":"YT",    "share(%)": round((alloc.get("YT",{}).get("share",0))*100,1),    "amount(¥)": alloc.get("YT",{}).get("amount",0)},
                {"media":"Tik",   "share(%)": round((alloc.get("Tik",{}).get("share",0))*100,1),   "amount(¥)": alloc.get("Tik",{}).get("amount",0)},
            ])
            st.dataframe(df_view, use_container_width=True)

            # ▼▼ ここから追加：理由・レポートの詳細をまとめて描画 ▼▼
            render_reasoning_and_report(res, sid)
            # ▲▲ ここまで追加 ▲▲

# スロット表示（3カラム / タブ 切替）
st.markdown("### スロットの表示方法")
display_mode = st.radio("スロットの表示方法", ["3カラム", "タブ"], horizontal=True, index=0)
if display_mode == "3カラム":
    cA, cB, cC = st.columns(3)
    with cA: slot_editor("A")
    with cB: slot_editor("B")
    with cC: slot_editor("C")
else:
    tabA, tabB, tabC = st.tabs(["A", "B", "C"])
    with tabA: slot_editor("A")
    with tabB: slot_editor("B")
    with tabC: slot_editor("C")

# -------------------------------------------------
# 横比較ビュー
# -------------------------------------------------
st.markdown("## 横比較（A/B/C）")
compare_rows = []
for sid in SLOT_IDS:
    cfg = st.session_state.slots[sid]
    res = cfg.get("result")
    if res:
        alloc = res.get("allocation", {})
        compare_rows.append({
            "slot": sid,
            "model": cfg.get("model"),
            "temp(指定?)": (cfg.get("temperature") if cfg.get("use_temperature") else "—"),
            "total(¥)": int(res.get("today_total_spend", 0)),
            "IGFB_share(%)": round(alloc.get("IGFB",{}).get("share",0)*100,1),
            "Google_share(%)": round(alloc.get("Google",{}).get("share",0)*100,1),
            "YT_share(%)": round(alloc.get("YT",{}).get("share",0)*100,1),
            "Tik_share(%)": round(alloc.get("Tik",{}).get("share",0)*100,1),
        })
if compare_rows:
    st.dataframe(pd.DataFrame(compare_rows), use_container_width=True)
else:
    st.info("まだ比較対象の結果がありません。各スロットで推論を実行してください。")

st.caption("※ BASE_INSTRUCTIONS を含め、全プロンプト/パラメータを UI から編集できます。任意パラメータは『指定する』ON時のみ API に送信します。")
