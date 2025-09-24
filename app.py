import json, time
import requests, pandas as pd, streamlit as st

st.set_page_config(page_title="AdAI 配分ビュー", layout="wide")

# ---------------------------
# セッション状態の初期化
# ---------------------------
if "src" not in st.session_state:
    st.session_state.src = None        # 取得した n8n JSON（加工後）
if "result" not in st.session_state:
    st.session_state.result = None     # LLMの最終配分JSON

# ---------------------------
# n8n 取得（手動実行のみ）
# ---------------------------
def _http_get_latest(timeout_s: int = 20):
    url = st.secrets["N8N_JSON_URL"]  # 例: https://yuya.app.n8n.cloud/webhook/latest
    auth = None
    if "N8N_BASIC_USER" in st.secrets and "N8N_BASIC_PASS" in st.secrets:
        auth = (st.secrets["N8N_BASIC_USER"], st.secrets["N8N_BASIC_PASS"])
    r = requests.get(url, auth=auth, timeout=timeout_s)
    r.raise_for_status()
    return r.json()

def pick_source(raw):
    # n8n が配列で返すケースも吸収（先頭を採用）
    if isinstance(raw, list):
        return raw[0] if raw and isinstance(raw[0], dict) else {}
    return raw if isinstance(raw, dict) else {}

def fetch_latest_manual(force: bool = False):
    """手動取得（必要ならキャッシュを明示クリアして再取得）"""
    if force:
        st.cache_data.clear()
    # 段階的リトライ（上限~70秒）
    timeouts = [10, 20, 40]
    last_err = None
    for i, t in enumerate(timeouts, start=1):
        try:
            raw = _http_get_latest(timeout_s=t)
            return pick_source(raw)
        except requests.exceptions.ReadTimeout as e:
            last_err = e
            time.sleep(0.8 * i)
    raise last_err or requests.exceptions.ReadTimeout("n8n webhook timeout")

# ---------------------------
# LLM（OpenAI / gpt-5）
# ---------------------------
BASE_INSTRUCTIONS = """あなたは広告予算配分の最適化アシスタントです。出力は厳密なJSONオブジェクトのみで返してください（余分な文章・コードフェンス不可）。

【目的】
- 月予算と4媒体（IGFB / Google / YT / Tik）の実績を踏まえ、成約単価（CPA）最適化の観点で「本日の総投資額」と「媒体別配分」を決定する。
- 月予算は使い切り必須ではない（効率悪化時は抑制可）。

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

【入力JSON】
"""

def build_prompt(src: dict) -> str:
    return BASE_INSTRUCTIONS + json.dumps(src, ensure_ascii=False, separators=(",", ":"))

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
    medias = ["IGFB", "Google", "YT", "Tik"]
    total = max(0, int(obj.get("today_total_spend", 0)))
    alloc = obj.get("allocation", {}) or {}
    for m in medias:
        if m not in alloc:
            alloc[m] = {"share": 0, "amount": 0}
        alloc[m]["share"] = round(float(alloc[m].get("share", 0) or 0), 2)
    # share合計=1.00補正
    ssum = round(sum(alloc[m]["share"] for m in medias), 2)
    if ssum == 0 and total > 0:
        for m in medias: alloc[m]["share"] = 0.25
        ssum = 1.0
    diff = round(1.00 - ssum, 2)
    if abs(diff) >= 0.01:
        maxm = max(medias, key=lambda m: alloc[m]["share"])
        alloc[maxm]["share"] = round(alloc[maxm]["share"] + diff, 2)
    # amount整合
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
st.title("AdAI 配分ビュー（手動取得 → 推論）")

with st.expander("手順", expanded=True):
    st.markdown("1) **データ取得** を押して n8n から最新JSONを取得\n2) 必要なら内容を確認\n3) **推論を実行** で gpt-5 による最終配分を生成")

cols = st.columns([1,1,4])
with cols[0]:
    if st.button("データ取得", type="primary"):
        with st.spinner("n8n から取得中…"):
            try:
                st.session_state.src = fetch_latest_manual(force=False)
                st.session_state.result = None
                st.success("取得成功")
            except requests.exceptions.ReadTimeout:
                st.error("n8n `/webhook/latest` がタイムアウトしました。")
            except requests.exceptions.HTTPError as e:
                st.error(f"HTTPエラー: {e}")
            except Exception as e:
                st.error(f"取得に失敗: {e}")
with cols[1]:
    if st.button("再取得（強制）"):
        with st.spinner("キャッシュを無視して再取得…"):
            try:
                st.session_state.src = fetch_latest_manual(force=True)
                st.session_state.result = None
                st.success("再取得成功")
            except Exception as e:
                st.error(f"再取得に失敗: {e}")

# 取得結果のプレビュー
src = st.session_state.src
if src:
    with st.expander("Raw（n8n返却JSONプレビュー）", expanded=False):
        st.code(json.dumps(src, ensure_ascii=False, indent=2))
else:
    st.info("まだデータを取得していません。上の **データ取得** ボタンを押してください。")

st.markdown("### 推論（OpenAI: gpt-5）")
if "OPENAI_API_KEY" not in st.secrets:
    st.warning("Secrets に OPENAI_API_KEY を設定してください。")
else:
    disabled = (src is None)
    if st.button("推論を実行", disabled=disabled):
        if disabled:
            st.warning("先にデータ取得を行ってください。")
        else:
            with st.spinner("gpt-5 で推論中…"):
                try:
                    prompt = BASE_INSTRUCTIONS + json.dumps(src, ensure_ascii=False, separators=(",", ":"))
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

# 推論結果の表示
res = st.session_state.result
if res:
    td = (src.get("facts") or {}).get("target_date") or res.get("report", {}).get("target_date", "")
    st.subheader(f"本日の配分（{td}）")
    st.metric("総額", f"¥{int(res.get('today_total_spend', 0)):,}")

    alloc = res.get("allocation", {})
    df = pd.DataFrame([
        {"media":"IGFB",  "share(%)": round(alloc["IGFB"]["share"]*100,1),  "amount(¥)": alloc["IGFB"]["amount"]},
        {"media":"Google","share(%)": round(alloc["Google"]["share"]*100,1),"amount(¥)": alloc["Google"]["amount"]},
        {"media":"YT",    "share(%)": round(alloc["YT"]["share"]*100,   1),"amount(¥)": alloc["YT"]["amount"]},
        {"media":"Tik",   "share(%)": round(alloc["Tik"]["share"]*100,  1),"amount(¥)": alloc["Tik"]["amount"]},
    ])
    st.dataframe(df, use_container_width=True)

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

st.caption("※ 初期表示ではWebhookへアクセスしません。必要時のみ手動取得 → 推論。")
