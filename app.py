# app.py
# AdAI 配分ビュー（facts + CSV 必須 / プロンプトA/B/Cを完全手動で比較）
# - 初期表示では外部アクセスしない
# - n8nの返却は「facts と csv.wide の両方」を必須
# - A/B/C それぞれで model / temperature / 他パラメータ / system&userプロンプト / CSV含有&期間 を独立設定
# - 各スロットの「最終投入プロンプト」「LLM生出力」「整合後JSON」を可視化し、横比較

import io, json, time, math
from datetime import datetime, timezone, timedelta, date
import requests, pandas as pd, streamlit as st

# ---------------------------
# 基本設定
# ---------------------------
st.set_page_config(page_title="AdAI 配分ビュー（facts+CSV必須 / A/B/C比較）", layout="wide")
JST = timezone(timedelta(hours=9))

# ---------------------------
# セッション状態
# ---------------------------
if "raw" not in st.session_state:      st.session_state.raw = None         # n8n返却そのまま
if "facts" not in st.session_state:    st.session_state.facts = None
if "csv_text" not in st.session_state: st.session_state.csv_text = None
if "meta" not in st.session_state:     st.session_state.meta = {}
if "df" not in st.session_state:       st.session_state.df = None
if "features" not in st.session_state: st.session_state.features = None    # LLMへ渡すベース（facts+channels）
if "slots" not in st.session_state:    st.session_state.slots = {}         # A/B/C 各設定・結果

SLOT_IDS = ["A", "B", "C"]

# ---------------------------
# n8n取得（手動のみ / BasicはSecretsがあれば自動）
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

# ---------------------------
# 入力正規化（facts + csv は必須）
#   受け入れ形：
#   - {"facts":{...}, "csv":{"wide":"..."}, "meta":{...}}
#   - [{"facts":{...}}, {"csv":{"wide":"..."}, "meta":{...}}]
# ---------------------------
def parse_wide_csv(csv_text: str) -> pd.DataFrame:
    df = pd.read_csv(io.StringIO(csv_text))
    if "variable" not in df.columns:
        raise ValueError("CSVに 'variable' 列がありません")
    df = df.set_index("variable")

    # 日付列を時系列順、非日付は末尾
    cols = []
    for c in df.columns:
        try:
            cols.append((pd.to_datetime(c), c))
        except Exception:
            cols.append((None, c))
    date_cols = sorted([c for d, c in cols if d is not None])
    other_cols = [c for d, c in cols if d is None]
    df = df[date_cols + other_cols]
    # 数値化（非日付は触らない）
    for c in date_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def coerce_require_facts_and_csv(raw):
    """facts と csv.wide を必須で抽出。無ければエラー。"""
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

    # channelsの所在（facts内 or トップ）に対応
    channels = {}
    if "channels" in facts and isinstance(facts["channels"], dict):
        channels = facts["channels"]
        facts = {k: v for k, v in facts.items() if k != "channels"}
    elif isinstance(raw, dict) and "channels" in raw and isinstance(raw["channels"], dict):
        channels = raw["channels"]

    return facts, channels, csv_text, meta

# ---------------------------
# CSVユーティリティ
# ---------------------------
def df_latest_days(df: pd.DataFrame, n_days: int) -> pd.DataFrame:
    cols = list(df.columns)
    # 末尾から n_days の日付列（非日付は除く）
    date_cols = [c for c in cols if str(c)[:4].isdigit()]
    use_cols = date_cols[-n_days:] if n_days > 0 else date_cols
    return df[use_cols]

def df_to_csv_text(df_slice: pd.DataFrame) -> str:
    # indexを 'variable' 列に戻してCSV文字列化
    out = df_slice.copy()
    out.insert(0, "variable", out.index)
    return out.to_csv(index=False)

# ---------------------------
# LLM 入出力関連
# ---------------------------
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

def build_features_for_prompt(facts: dict, channels: dict, override_budget: float|None, override_target: date|None):
    """factsにUIの上書きを適用し、featuresにまとめる"""
    f = dict(facts)  # copy
    if override_budget is not None:
        f["month_budget"] = float(override_budget)
    if override_target is not None:
        f["target_date"] = override_target.strftime("%Y-%m-%d")
    return {"facts": f, "channels": channels}

def make_final_prompt_text(system_text: str, user_text: str, features_obj: dict, include_csv: bool, csv_snippet: str|None):
    base = BASE_INSTRUCTIONS + json.dumps(features_obj, ensure_ascii=False, separators=(",", ":"))
    if include_csv and csv_snippet:
        base += "\n\n【CSV（最新N日）】\n" + csv_snippet
    # user_text を最後に付与（追加の指示や評価軸の変更など）
    if user_text and user_text.strip():
        base += "\n\n【追加指示】\n" + user_text.strip()
    # system_text は OpenAIの system に渡す。ここでは返却としても見えるようにしておく
    visible_prompt = f"<<SYSTEM>>\n{system_text.strip()}\n\n<<USER>>\n{base}"
    return base, visible_prompt  # base: 実際にuserへ、visible: UI表示用

def run_openai_chat(slot_cfg: dict, user_prompt: str) -> str:
    """SDK優先 → HTTPフォールバック。JSON文字列を返す（response_format=json_object）"""
    api_key = st.secrets["OPENAI_API_KEY"]
    base_url = st.secrets.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    model = slot_cfg["model"]
    temperature = float(slot_cfg["temperature"])
    top_p = slot_cfg.get("top_p")
    max_tokens = slot_cfg.get("max_tokens")
    seed = slot_cfg.get("seed")
    presence_penalty = slot_cfg.get("presence_penalty")
    frequency_penalty = slot_cfg.get("frequency_penalty")
    system_text = slot_cfg["system"]

    messages = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": user_prompt},
    ]

    # 1) SDK
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url=base_url)
        kwargs = {
            "model": model,
            "response_format": {"type": "json_object"},
            "temperature": temperature,
            "messages": messages,
        }
        if top_p is not None: kwargs["top_p"] = float(top_p)
        if max_tokens is not None: kwargs["max_tokens"] = int(max_tokens)
        if seed is not None: kwargs["seed"] = int(seed)
        if presence_penalty is not None: kwargs["presence_penalty"] = float(presence_penalty)
        if frequency_penalty is not None: kwargs["frequency_penalty"] = float(frequency_penalty)

        resp = client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content
    except ModuleNotFoundError:
        pass

    # 2) HTTP フォールバック
    body = {
        "model": model,
        "response_format": {"type": "json_object"},
        "temperature": temperature,
        "messages": messages,
    }
    if top_p is not None: body["top_p"] = float(top_p)
    if max_tokens is not None: body["max_tokens"] = int(max_tokens)
    if seed is not None: body["seed"] = int(seed)
    if presence_penalty is not None: body["presence_penalty"] = float(presence_penalty)
    if frequency_penalty is not None: body["frequency_penalty"] = float(frequency_penalty)

    r = requests.post(
        f"{base_url}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=body, timeout=120
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

# ---------------------------
# UI: ヘッダ
# ---------------------------
st.title("AdAI 配分ビュー（facts+CSV 必須 / A-B-C プロンプト比較）")
st.caption(f"現在時刻（JST）: {datetime.now(JST).strftime('%Y-%m-%d %H:%M:%S')}")

# ---------------------------
# サイドバー：共通設定
# ---------------------------
st.sidebar.header("共通設定")
st.sidebar.markdown("- まず「データ取得」を押して n8n から facts と CSV を読み込みます。")
if "OPENAI_API_KEY" not in st.secrets:
    st.sidebar.error("Secrets に OPENAI_API_KEY を設定してください。")

# facts の上書き
override_budget = st.sidebar.number_input("月予算（上書き / 任意）", min_value=0.0, step=1000.0, format="%.0f", value=0.0)
use_override_budget = st.sidebar.checkbox("月予算の上書きを有効化", value=False)

override_target_date = st.sidebar.date_input(
    "対象日（上書き / 任意）", value=None, min_value=date(2000,1,1), max_value=date(2100,12,31), format="YYYY-MM-DD"
)
use_override_target = st.sidebar.checkbox("対象日の上書きを有効化", value=False)

# CSVをプロンプトに含める既定（スロット毎に別設定も可能）
st.sidebar.divider()
st.sidebar.write("**CSV同梱の既定（各スロットでも変更可）**")
csv_default_include = st.sidebar.checkbox("CSVをプロンプトへ含める（既定）", value=True)
csv_default_days = st.sidebar.slider("CSVの最新N日（既定）", 3, 31, 7, 1)

# ---------------------------
# 取得 / 再取得
# ---------------------------
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

                # features（プロンプト用ベース）は上書き適用時に都度作るのでここではNone
                st.session_state.features = None

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
                st.session_state.features = None
                st.success("再取得成功")
            except Exception as e:
                st.error(f"再取得に失敗: {e}")

with c3:
    st.info("注意：facts と CSV の両方が無いと推論できません。")

# ---------------------------
# プレビュー：CSV & facts
# ---------------------------
if st.session_state.df is not None:
    st.subheader("CSVプレビュー（全期間 / 先頭10行）")
    st.dataframe(st.session_state.df.head(10), use_container_width=True)

    st.subheader("CSVプレビュー（最新7日 / 先頭10行）")
    try:
        st.dataframe(df_latest_days(st.session_state.df, 7).head(10), use_container_width=True)
    except Exception:
        st.info("最新7日抽出に失敗しましたが全体は読み込めています。")

if st.session_state.facts is not None:
    st.subheader("facts（元データ）")
    st.code(json.dumps(st.session_state.facts, ensure_ascii=False, indent=2))

# ---------------------------
# スロット初期化（A/B/C）
# ---------------------------
def ensure_slots():
    defaults = {
        "model": st.secrets.get("OPENAI_MODEL", "gpt-4o-mini"),
        "temperature": 0.2,
        "top_p": None,
        "max_tokens": None,
        "seed": None,
        "presence_penalty": None,
        "frequency_penalty": None,
        "system": "You are an expert ads budget allocation assistant. Output valid JSON only.",
        "user": "数値根拠を明示しつつ、現実的な配分を提案してください。",
        "include_csv": csv_default_include,
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

# ---------------------------
# スロット編集UI
# ---------------------------
st.markdown("## Prompt Lab（A/B/C）— モデル/温度/プロンプト等を完全手動で設定")

def slot_editor(sid: str):
    cfg = st.session_state.slots[sid]
    with st.expander(f"スロット {sid} の設定・実行", expanded=(sid=="A")):
        c1, c2, c3, c4 = st.columns([1.2,1,1,1])
        with c1:
            cfg["model"] = st.text_input(f"[{sid}] モデル名", value=cfg["model"], key=f"{sid}_model")
        with c2:
            cfg["temperature"] = st.number_input(f"[{sid}] temperature", min_value=0.0, max_value=2.0, step=0.1, value=float(cfg["temperature"]), key=f"{sid}_temp")
        with c3:
            cfg["top_p"] = st.number_input(f"[{sid}] top_p（任意）", min_value=0.0, max_value=1.0, step=0.05, value=cfg["top_p"] if cfg["top_p"] is not None else 1.0, key=f"{sid}_top_p")
            if math.isclose(cfg["top_p"], 1.0): cfg["top_p"] = None  # 1.0は未指定扱い
        with c4:
            cfg["max_tokens"] = st.number_input(f"[{sid}] max_tokens（任意）", min_value=0, step=50, value=cfg["max_tokens"] or 0, key=f"{sid}_max_tok")
            if cfg["max_tokens"] == 0: cfg["max_tokens"] = None

        c5, c6, c7 = st.columns(3)
        with c5:
            cfg["seed"] = st.number_input(f"[{sid}] seed（任意）", min_value=0, step=1, value=cfg["seed"] or 0, key=f"{sid}_seed")
            if cfg["seed"] == 0: cfg["seed"] = None
        with c6:
            cfg["presence_penalty"] = st.number_input(f"[{sid}] presence_penalty（任意）", min_value=-2.0, max_value=2.0, step=0.1, value=cfg["presence_penalty"] if cfg["presence_penalty"] is not None else 0.0, key=f"{sid}_pp")
            if math.isclose(cfg["presence_penalty"], 0.0): cfg["presence_penalty"] = None
        with c7:
            cfg["frequency_penalty"] = st.number_input(f"[{sid}] frequency_penalty（任意）", min_value=-2.0, max_value=2.0, step=0.1, value=cfg["frequency_penalty"] if cfg["frequency_penalty"] is not None else 0.0, key=f"{sid}_fp")
            if math.isclose(cfg["frequency_penalty"], 0.0): cfg["frequency_penalty"] = None

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
                        # 上書きを適用した features
                        ov_budget = float(override_budget) if use_override_budget else None
                        ov_target = override_target_date if use_override_target else None
                        channels = st.session_state.raw[0].get("facts", {}).get("channels") if isinstance(st.session_state.raw, list) else {}
                        # channelsは coerce で分離済みのため、セッション上にはない可能性があるので再構成
                        # → 一貫のため、coerce時のchannelsを featuresに持たせる:
                        #   st.session_state.features は実行時に都度作る
                        facts, chs = st.session_state.facts, {}
                        # coerceで channels を facts から分離した前提。rawに戻るのが難しいため、featuresは facts単体でもOK。
                        # もし facts 内に channels が残っていれば拾う
                        if "channels" in facts and isinstance(facts["channels"], dict):
                            chs = facts["channels"]

                        features = build_features_for_prompt(facts, chs, ov_budget, ov_target)

                        # CSV断片
                        csv_snippet = None
                        if cfg["include_csv"]:
                            try:
                                df_slice = df_latest_days(st.session_state.df, int(cfg["csv_days"]))
                                csv_snippet = df_to_csv_text(df_slice)
                            except Exception as e:
                                csv_snippet = None
                                st.warning(f"CSV断片の生成に失敗: {e}")

                        # 最終プロンプト生成
                        user_prompt, visible_prompt = make_final_prompt_text(
                            cfg["system"], cfg["user"], features, cfg["include_csv"], csv_snippet
                        )
                        cfg["final_prompt_preview"] = visible_prompt

                        # OpenAI 呼び出し
                        raw_text = run_openai_chat(cfg, user_prompt)
                        cfg["raw_output"] = raw_text

                        # JSON整形
                        try:
                            obj = json.loads(raw_text.strip().strip("`"))
                        except json.JSONDecodeError:
                            # フェンス付きなどの緊急除去
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

        # プレビュー（最終プロンプト / 生出力 / 整合後結果）
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
            if res.get("report", {}).get("executive_summary"):
                st.write("**要約**")
                st.write(res["report"]["executive_summary"])

# スロットUI描画
cA, cB, cC = st.columns(3)
with cA: slot_editor("A")
with cB: slot_editor("B")
with cC: slot_editor("C")

# ---------------------------
# 横比較ビュー
# ---------------------------
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
            "temp": cfg.get("temperature"),
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

st.caption("※ facts と CSV は必須。各スロットでモデル/温度/プロンプト等を独立に変更し、最終プロンプトと生出力も確認できます。")
