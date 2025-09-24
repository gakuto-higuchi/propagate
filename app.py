import json, math
import requests, pandas as pd, streamlit as st

st.set_page_config(page_title="AdAI 配分ビュー", layout="wide")

# ------------ 取得系 ------------
@st.cache_data(ttl=300)
def fetch_latest():
    """n8n Webhook から最新データを取得（Bearerなし。BasicはSecretsにあれば使う）"""
    url = st.secrets["N8N_JSON_URL"]
    auth = None
    user, pwd = st.secrets.get("N8N_BASIC_USER"), st.secrets.get("N8N_BASIC_PASS")
    if user and pwd:
        auth = (user, pwd)  # Basic認証が必要な場合のみ
    r = requests.get(url, auth=auth, timeout=20)
    r.raise_for_status()
    return r.json()

def pick_source(raw):
    """配列/オブジェクトどちらでも '解析に使う元データ' を返す"""
    if isinstance(raw, list):
        return raw[0] if raw else {}
    return raw

# ------------ LLMプロンプト ------------
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
  "reasoning_points": [
    "判断ポイントを短文で（数値根拠も簡潔に）",
    "…"
  ],
  "report": {
    "title": "本日の予算配分レポート",
    "target_date": "<YYYY-MM-DD>",
    "executive_summary": "一段落で要旨",
    "pacing_decision": {
      "chosen_today_total": <int>,
      "why": "参照値と判断根拠を数値で説明"
    },
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
    "risks_and_actions": {
      "risks": ["…","…"],
      "next_actions": ["…","…"]
    }
  }
}

【入力JSON】
これを解析に使ってください（そのまま引用可、個人情報なし）：
"""

def build_prompt(src: dict) -> str:
    return BASE_INSTRUCTIONS + json.dumps(src, ensure_ascii=False, separators=(',', ':'))

# ------------ LLM呼び出し ------------
def call_openai_chat(prompt: str) -> str:
    """OpenAI Chat Completions API を直接呼ぶ（gpt-5, bearerはSecretsのAPIキー）"""
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
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    r = requests.post(f"{base_url}/chat/completions", headers=headers, json=body, timeout=120)
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"]

def strip_code_fence(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        # ```json ... ``` or ``` ...
        s = s.split("```", 2)[1] if s.count("```") >= 2 else s
        s = s.replace("json", "", 1).strip()
    return s.strip("` \n\r\t")

def enforce_constraints(obj: dict) -> dict:
    """share小数2桁で合計1.00、amount合計=total を保証（最大share媒体で差分吸収）"""
    alloc = obj.get("allocation", {})
    total = int(max(0, int(obj.get("today_total_spend", 0))))
    # シェア丸め（2桁）
    medias = ["IGFB", "Google", "YT", "Tik"]
    for m in medias:
        if m not in alloc:
            alloc[m] = {"share": 0, "amount": 0}
        alloc[m]["share"] = round(float(alloc[m].get("share", 0) or 0), 2)

    # 合計1.00に補正
    shares = [alloc[m]["share"] for m in medias]
    ssum = round(sum(shares), 2)
    if ssum == 0 and total > 0:
        # 全ゼロは均等割（保険）
        for m in medias: alloc[m]["share"] = 0.25
        ssum = 1.0
    diff = round(1.00 - ssum, 2)
    if abs(diff) >= 0.01:
        maxm = max(medias, key=lambda m: alloc[m]["share"])
        alloc[maxm]["share"] = round(alloc[maxm]["share"] + diff, 2)

    # amount再計算（四捨五入→差分は最大share媒体へ）
    amounts = [int(round(total * alloc[m]["share"])) for m in medias]
    adiff = total - sum(amounts)
    if adiff != 0:
        maxm = max(medias, key=lambda m: alloc[m]["share"])
        idx = medias.index(maxm)
        amounts[idx] += adiff
    for i, m in enumerate(medias):
        alloc[m]["amount"] = max(0, int(amounts[i]))

    obj["today_total_spend"] = total
    obj["allocation"] = alloc
    return obj

# ------------ UI ------------
raw = fetch_latest()
src = pick_source(raw)

st.title("AdAI 配分ビュー（n8n→SCC）")
with st.expander("Raw（n8n返却JSON）", expanded=False):
    st.code(json.dumps(src, ensure_ascii=False, indent=2))

# 推論ボタン
st.markdown("### 推論（OpenAI: gpt-5 で最終配分を生成）")
col_run, col_model = st.columns([1, 2])
with col_model:
    st.caption("モデルやエンドポイントは Secrets で変更可（`OPENAI_MODEL`, `OPENAI_BASE_URL`）")

if st.button("推論を実行（gpt-5）", type="primary"):
    try:
        prompt = build_prompt(src)
        out_text = call_openai_chat(prompt)
        out_text = strip_code_fence(out_text)
        result = json.loads(out_text)  # 厳密JSON前提
        result = enforce_constraints(result)

        # 表示
        st.success("推論に成功")
        target_date = (src.get("facts") or {}).get("target_date") or result.get("report", {}).get("target_date", "")
        st.subheader(f"本日の配分（{target_date}）")
        st.metric("総額", f"¥{int(result.get('today_total_spend', 0)):,}")

        alloc = result.get("allocation", {})
        df = pd.DataFrame([
            {"media": "IGFB",  "share(%)": round(alloc["IGFB"]["share"]*100, 1),  "amount(¥)": alloc["IGFB"]["amount"]},
            {"media": "Google","share(%)": round(alloc["Google"]["share"]*100,1),"amount(¥)": alloc["Google"]["amount"]},
            {"media": "YT",    "share(%)": round(alloc["YT"]["share"]*100,   1),"amount(¥)": alloc["YT"]["amount"]},
            {"media": "Tik",   "share(%)": round(alloc["Tik"]["share"]*100,  1),"amount(¥)": alloc["Tik"]["amount"]},
        ])
        st.dataframe(df, use_container_width=True)

        if result.get("reasoning_points"):
            st.markdown("#### 判断ポイント")
            for p in result["reasoning_points"]:
                st.write("• " + str(p))

        if result.get("report"):
            st.markdown("#### 要約")
            st.write(result["report"].get("executive_summary",""))

        # ダウンロード
        st.download_button("結果JSONをダウンロード",
                           data=json.dumps(result, ensure_ascii=False, indent=2),
                           file_name="allocation_result.json",
                           mime="application/json")
    except json.JSONDecodeError as e:
        st.error(f"LLM出力がJSONとして解釈できませんでした: {e}")
        st.text(out_text[:1000])
    except Exception as e:
        st.error(f"推論に失敗: {e}")

st.divider()
st.caption("※ n8n Webhookは軽量化、重い解析はSCC側で実施。APIキー/URLはSecrets管理。")
