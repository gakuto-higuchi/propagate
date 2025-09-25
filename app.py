# app.py
# AdAI 配分ビュー（facts + CSV 必須 / A-B-C プロンプト比較）
# - すべての生成パラメータは「任意」。未指定のものは API に含めない。
# - 初期表示では外部アクセスしない
# - n8n の返却は「facts と csv.wide の両方」を必須
# - A/B/C で model / temperature / top_p / max_tokens / seed /
#   presence_penalty / frequency_penalty / system & user prompt / CSV同梱 を独立設定
# - 各スロットの「最終投入プロンプト」「LLM生出力」「整合後JSON」を可視化

import io, json, time, math
from datetime import datetime, timezone, timedelta, date
import requests, pandas as pd, streamlit as st

# =========================
# 基本設定 + CSS
# =========================
st.set_page_config(page_title="AdAI 配分ビュー（facts+CSV 必須 / A-B-C 比較）", layout="wide")
JST = timezone(timedelta(hours=9))

def inject_css(compact: bool = False):
    # タイトルが切れないよう上余白を確保
    padding_top = "2.75rem" if not compact else "1.75rem"
    css = f"""
    <style>
    .block-container {{
        max-width: 1400px;
        padding-top: {padding_top};
        padding-bottom: 2rem;
        scroll-padding-top: 80px;
    }}
    .block-container h1:first-of-type {{
        margin-top: 0.25rem !important;
        padding-top: 0.25rem !important;
        line-height: 1.2 !important;
        overflow: visible !important;
    }}
    [data-testid="stDataFrame"] .ag-header-cell-label {{ white-space: nowrap !important; }}
    .stButton>button {{ width: 100%; }}
    .st-expander {{ border: 1px solid #e5e7eb; border-radius: 12px; }}
    div.row-widget.stRadio > div {{ flex-wrap: wrap; gap: .5rem 1rem; }}
    pre, code {{ white-space: pre; overflow-x: auto; }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# =========================
# セッション状態
# =========================
if "raw" not in st.session_state:      st.session_state.raw = None
if "facts" not in st.session_state:    st.session_state.facts = None
if "csv_text" not in st.session_state: st.session_state.csv_text = None
if "meta" not in st.session_state:     st.session_state.meta = {}
if "df" not in st.session_state:       st.session_state.df = None
if "slots" not in st.session_state:    st.session_state.slots = {}

SLOT_IDS = ["A", "B", "C"]

# =========================
# n8n 取得（手動のみ）
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
            time.sleep(0.7 * i)
    raise last_err or requests.exceptions.ReadTimeout("n8n webhook timeout")

# =========================
# 入力正規化（facts + csv は必須）
# 受け入れ：{"facts":{...}, "csv":{"wide":"..."}, "meta":{...}}
#       or [{"facts":{...}}, {"csv":{"wide":"..."}, "meta":{...}}]
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

def coerce_require_facts_and_csv(raw):
    facts = None
    csv_text = None
    meta = {}
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict):
                if facts is None and "facts" in item:
                    facts = item["facts"]
                if csv_text is None and isinstance(item.get("csv"), dict) and "wide" in item["csv"]:
                    csv_text = item["csv"]["wide"]
                    meta = item.get("meta", {}) or meta
    elif isinstance(raw, dict):
        if "facts" in raw:
            facts = raw["facts"]
        if isinstance(raw.get("csv"), dict) and "wide" in raw["csv"]:
            csv_text = raw["csv"]["wide"]
            meta = raw.get("meta", {}) or meta
    else:
        raise ValueError("未対応のペイロード形式です（list/dict 以外）")

    if not isinstance(facts, dict):
        raise ValueError("facts が見つかりませんでした（必須）")
    if not csv_text or not isinstance(csv_text, str):
        raise ValueError("csv.wide が見つかりませんでした（必須）")

    # facts 内の channels を分離（無くてもOK）
    if "channels" in facts and isinstance(facts["channels"], dict):
        # 今回はそのまま facts 内に保持して渡しても良い
        pass

    return facts, facts.get("channels", {}), csv_text, meta

# =========================
# CSV ユーティリティ
# =========================
def df_latest_days(df: pd.DataFrame, n_days: int) -> pd.DataFrame:
    cols = list(df.columns)
    date_cols = [c for c in cols if str(c)[:4].isdigit()]
    use_cols = date_cols[-n_days:] if n_days > 0 else date_cols
    return df[use_cols]

def df_to_csv_text(df_slice: pd.DataFrame) -> str:
    out = df_slice.copy()
    out.insert(0, "variable", out.index)
    return out.to_csv(index=False)

# =========================
# LLM 入出力
# =========================
BASE_INSTRUCTIONS = """あなたは広告予算配分の最適化アシスタントです。出力は厳密なJSONオブジェクトのみで返してください（余分な文章・コードフェンス不可）。

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
{ "today_total_spend": <int>,
  "allocation": {
    "IGFB":   { "share": <number>, "amount": <int> },
    "Google": { "share": <number>, "amount": <int> },
    "YT":     { "share": <number>, "amount": <int> },
    "Tik":    { "share": <number>, "amount": <int> }
  },
  "reasoning_points": ["…","…"],
  "report": { ... }
}

【入力（features: facts/channels）】
"""

def build_features_for_prompt(facts: dict, channels: dict, override_budget: float|None, override_target: date|None):
    f = dict(facts)
    if override_budget is not None:
        f["month_budget"] = float(override_budget)
    if override_target is not None:
        f["target_date"] = override_target.strftime("%Y-%m-%d")
    return {"facts": f, "channels": channels}

def make_final_prompt_text(system_text: str, user_text: str, features_obj: dict, include_csv: bool, csv_snippet: str|None):
    base = BASE_INSTRUCTIONS + json.dumps(features_obj, ensure_ascii=False, separators=(",", ":"))
    if include_csv and csv_snippet:
        base += "\n\n【CSV（最新N日）】\n" + csv_snippet
    if user_text and user_text.strip():
        base += "\n\n【追加指示】\n" + user_text.strip()
    visible_prompt = f"<<SYSTEM>>\n{system_text.strip()}\n\n<<USER>>\n{base}"
    return base, visible_prompt

def run_openai_chat(slot_cfg: dict, user_prompt: str) -> str:
    """SDK優先 → HTTPフォールバック。未指定パラメータは送信しない"""
    api_key = st.secrets["OPENAI_API_KEY"]
    base_url = st.secrets.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    model = slot_cfg["model"]
    system_text = slot_cfg["system"]

    messages = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": user_prompt},
    ]

    # 共通の可変パラメータ組み立て（Noneは入れない）
    def build_params():
        p = {
            "model": model,
            "response_format": {"type": "json_object"},
            "messages": messages,
        }
        if slot_cfg.get("use_temperature") and slot_cfg.get("temperature") is not None:
            p["temperature"] = float(slot_cfg["temperature"])
        if slot_cfg.get("use_top_p") and slot_cfg.get("top_p") is not None:
            p["top_p"] = float(slot_cfg["top_p"])
        if slot_cfg.get("use_max_tokens") and slot_cfg.get("max_tokens") is not None and int(slot_cfg["max_tokens"]) > 0:
            p["max_tokens"] = int(slot_cfg["max_tokens"])
        if slot_cfg.get("use_seed") and slot_cfg.get("seed") is not None:
            p["seed"] = int(slot_cfg["seed"])
        if slot_cfg.get("use_presence_penalty") and slot_cfg.get("presence_penalty") is not None:
            p["presence_penalty"] = float(slot_cfg["presence_penalty"])
        if slot_cfg.get("use_frequency_penalty") and slot_cfg.get("frequency_penalty") is not None:
            p["frequency_penalty"] = float(slot_cfg["frequency_penalty"])
        return p

    # 1) SDK
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url=base_url)
        resp = client.chat.completions.create(**build_params())
        return resp.choices[0].message.content
    except ModuleNotFoundError:
        pass

    # 2) HTTP フォールバック
    r = requests.post(
        f"{base_url}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=build_params(), timeout=120
    )
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"]

def enforce_constraints(obj: dict) -> dict:
    medias = ["IGFB", "Google", "YT", "Tik"]
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

# =========================
# UI: ヘッダ & サイドバー
# =========================
inject_css(compact=False)
st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
st.title("AdAI 配分ビュー（facts+CSV 必須 / A-B-C プロンプト比較）")
st.caption(f"現在時刻（JST）: {datetime.now(JST).strftime('%Y-%m-%d %H:%M:%S')}")

st.sidebar.header("共通設定")
compact = st.sidebar.checkbox("コンパクト表示（余白を減らす）", value=False, help="見切れが出る場合はOFF推奨")
if compact:
    inject_css(compact=True)

if "OPENAI_API_KEY" not in st.secrets:
    st.sidebar.error("Secrets に OPENAI_API_KEY を設定してください。")

override_budget = st.sidebar.number_input("月予算（上書き / 任意）", min_value=0.0, step=1000.0, format="%.0f", value=0.0)
use_override_budget = st.sidebar.checkbox("月予算の上書きを有効化", value=False)

override_target_date = st.sidebar.date_input("対象日（上書き / 任意）", value=None, min_value=date(2000,1,1), max_value=date(2100,12,31), format="YYYY-MM-DD")
use_override_target = st.sidebar.checkbox("対象日の上書きを有効化", value=False)

st.sidebar.divider()
st.sidebar.write("**CSV同梱の既定（各スロットでも変更可）**")
csv_default_include = st.sidebar.checkbox("CSVをプロンプトへ含める（既定）", value=True)
csv_default_days = st.sidebar.slider("CSVの最新N日（既定）", 3, 31, 7, 1)

# =========================
# 取得 / 再取得
# =========================
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
                st.session_state.csv_text = csv_text
                st.session_state.meta = meta
                st.session_state.df = df
                st.success("取得成功：facts と CSV を読み込みました。")
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
                st.session_state.csv_text = csv_text
                st.session_state.meta = meta
                st.session_state.df = df
                st.success("再取得成功")
            except Exception as e:
                st.error(f"再取得に失敗: {e}")

with c3:
    st.info("注意：facts と CSV の両方が無いと推論できません。")

# =========================
# プレビュー：CSV & facts
# =========================
if st.session_state.df is not None:
    st.subheader("CSVプレビュー（全期間・全行）")
    st.dataframe(st.session_state.df, use_container_width=True, height=600)

if st.session_state.facts is not None:
    st.subheader("facts（元データ）")
    st.code(json.dumps(st.session_state.facts, ensure_ascii=False, indent=2))

# =========================
# スロット初期化
# =========================
def ensure_slots():
    defaults = {
        "model": st.secrets.get("OPENAI_MODEL", "gpt-4o-mini"),
        # 任意パラメータ：デフォルトは「送らない」
        "use_temperature": False,
        "temperature": 1.0,
        "use_top_p": False,
        "top_p": 1.0,
        "use_max_tokens": False,
        "max_tokens": 0,
        "use_seed": False,
        "seed": 0,
        "use_presence_penalty": False,
        "presence_penalty": 0.0,
        "use_frequency_penalty": False,
        "frequency_penalty": 0.0,
        # プロンプト & CSV
        "system": "You are an expert ads budget allocation assistant. Output valid JSON only.",
        "user": "数値根拠を明示しつつ、現実的な配分を提案してください。",
        "include_csv": csv_default_include,
        "csv_days": csv_default_days,
        # 結果
        "final_prompt_preview": None,
        "raw_output": None,
        "result": None,
    }
    for sid in SLOT_IDS:
        if sid not in st.session_state.slots:
            st.session_state.slots[sid] = dict(defaults)

ensure_slots()

# =========================
# スロット編集UI
# =========================
st.markdown("## Prompt Lab（A/B/C）— モデル/温度/プロンプト等を完全手動で設定")

def slot_editor(sid: str):
    cfg = st.session_state.slots[sid]
    with st.expander(f"スロット {sid} の設定・実行", expanded=(sid=="A")):
        c1, c2, c3, c4 = st.columns([1.4,1.1,1.1,1.1])
        with c1:
            cfg["model"] = st.text_input(f"[{sid}] モデル名", value=cfg["model"], key=f"{sid}_model")

        with c2:
            cfg["use_temperature"] = st.checkbox(f"[{sid}] temperature を指定", value=cfg["use_temperature"], key=f"{sid}_use_temp")
            if cfg["use_temperature"]:
                cfg["temperature"] = st.number_input(f"[{sid}] temperature 値", min_value=0.0, max_value=2.0, step=0.1, value=float(cfg["temperature"]), key=f"{sid}_temp_val")

        with c3:
            cfg["use_top_p"] = st.checkbox(f"[{sid}] top_p を指定", value=cfg["use_top_p"], key=f"{sid}_use_top_p")
            if cfg["use_top_p"]:
                cfg["top_p"] = st.number_input(f"[{sid}] top_p 値", min_value=0.0, max_value=1.0, step=0.05, value=float(cfg["top_p"]), key=f"{sid}_top_p_val")

        with c4:
            cfg["use_max_tokens"] = st.checkbox(f"[{sid}] max_tokens を指定", value=cfg["use_max_tokens"], key=f"{sid}_use_max_tokens")
            if cfg["use_max_tokens"]:
                cfg["max_tokens"] = st.number_input(f"[{sid}] max_tokens 値", min_value=1, step=50, value=int(cfg["max_tokens"] or 256), key=f"{sid}_max_tok_val")

        c5, c6, c7 = st.columns([1,1,1])
        with c5:
            cfg["use_seed"] = st.checkbox(f"[{sid}] seed を指定", value=cfg["use_seed"], key=f"{sid}_use_seed")
            if cfg["use_seed"]:
                cfg["seed"] = st.number_input(f"[{sid}] seed 値", min_value=0, step=1, value=int(cfg["seed"] or 0), key=f"{sid}_seed_val")
        with c6:
            cfg["use_presence_penalty"] = st.checkbox(f"[{sid}] presence_penalty を指定", value=cfg["use_presence_penalty"], key=f"{sid}_use_pp")
            if cfg["use_presence_penalty"]:
                cfg["presence_penalty"] = st.number_input(f"[{sid}] presence_penalty 値", min_value=-2.0, max_value=2.0, step=0.1, value=float(cfg["presence_penalty"] or 0.0), key=f"{sid}_pp_val")
        with c7:
            cfg["use_frequency_penalty"] = st.checkbox(f"[{sid}] frequency_penalty を指定", value=cfg["use_frequency_penalty"], key=f"{sid}_use_fp")
            if cfg["use_frequency_penalty"]:
                cfg["frequency_penalty"] = st.number_input(f"[{sid}] frequency_penalty 値", min_value=-2.0, max_value=2.0, step=0.1, value=float(cfg["frequency_penalty"] or 0.0), key=f"{sid}_fp_val")

        cfg["include_csv"] = st.checkbox(f"[{sid}] CSVをプロンプトへ含める", value=cfg["include_csv"], key=f"{sid}_incl_csv")
        if cfg["include_csv"]:
            cfg["csv_days"] = st.slider(f"[{sid}] CSVの最新N日", 3, 31, cfg["csv_days"], 1, key=f"{sid}_csv_days")

        cfg["system"] = st.text_area(f"[{sid}] System プロンプト", value=cfg["system"], height=100, key=f"{sid}_sys")
        cfg["user"]   = st.text_area(f"[{sid}] User プロンプト（追加指示）", value=cfg["user"], height=140, key=f"{sid}_usr")

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
                        facts, channels = st.session_state.facts, {}
                        if "channels" in facts and isinstance(facts["channels"], dict):
                            channels = facts["channels"]
                        features = build_features_for_prompt(facts, channels, ov_budget, ov_target)

                        csv_snippet = None
                        if cfg["include_csv"]:
                            try:
                                df_slice = df_latest_days(st.session_state.df, int(cfg["csv_days"]))
                                csv_snippet = df_to_csv_text(df_slice)
                            except Exception as e:
                                st.warning(f"CSV断片の生成に失敗: {e}")

                        user_prompt, visible_prompt = make_final_prompt_text(cfg["system"], cfg["user"], features, cfg["include_csv"], csv_snippet)
                        cfg["final_prompt_preview"] = visible_prompt

                        raw_text = run_openai_chat(cfg, user_prompt)
                        cfg["raw_output"] = raw_text

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

        # プレビュー
        if cfg.get("final_prompt_preview"):
            st.markdown(f"**[{sid}] 最終プロンプト（実投入内容の可視化）**")
            st.code(cfg["final_prompt_preview"])
        if cfg.get("raw_output") is not None:
            st.markdown(f"**[{sid}] LLM 生出力（加工前）**")
            st.code(cfg["raw_output"])
        if cfg.get("result"):
            res = cfg["result"]
            st.markdown(f"**[{sid}] 整合後の配分結果**")
            st.metric(f"[{sid}] 総額", f"¥{int(res.get('today_total_spend', 0)):,}")
            alloc = res.get("allocation", {})
            df_view = pd.DataFrame([
                {"media":"IGFB",  "share(%)": round((alloc.get("IGFB",{}).get("share",0))*100,1),  "amount(¥)": alloc.get("IGFB",{}).get("amount",0)},
                {"media":"Google","share(%)": round((alloc.get("Google",{}).get("share",0))*100,1),"amount(¥)": alloc.get("Google",{}).get("amount",0)},
                {"media":"YT",    "share(%)": round((alloc.get("YT",{}).get("share",0))*100,1),    "amount(¥)": alloc.get("YT",{}).get("amount",0)},
                {"media":"Tik",   "share(%)": round((alloc.get("Tik",{}).get("share",0))*100,1),   "amount(¥)": alloc.get("Tik",{}).get("amount",0)},
            ])
            st.dataframe(df_view, use_container_width=True)
            if res.get("report", {}).get("executive_summary"):
                st.write("**要約**")
                st.write(res["report"]["executive_summary"])

# 描画
cA, cB, cC = st.columns(3)
with cA: slot_editor("A")
with cB: slot_editor("B")
with cC: slot_editor("C")

# =========================
# 横比較ビュー
# =========================
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
            "temp": (cfg.get("temperature") if cfg.get("use_temperature") else "(default)"),
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

st.caption("※ 未指定の生成パラメータは API に送信しません。モデルによる制約で 400 になるのを防ぎます。")
