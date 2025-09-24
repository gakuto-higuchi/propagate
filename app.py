import streamlit as st, requests, pandas as pd

st.set_page_config(page_title="AdAI 配分ビュー", layout="wide")

@st.cache_data(ttl=300)
def fetch_latest():
    url = st.secrets["N8N_JSON_URL"]      # 例: https://<your-n8n>/webhook/ad-ai/latest
    headers = {}
    token = st.secrets.get("N8N_BEARER")  # WebhookをBearerで保護してる場合のみ
    if token:
        headers["Authorization"] = f"Bearer {token}"
    r = requests.get(url, headers=headers, timeout=15)
    r.raise_for_status()
    return r.json()

data = fetch_latest()
payload = data.get("payload", data)
target_date = data.get("target_date") or payload.get("target_date")

st.title(f"本日の配分（{target_date or ''}）")
st.caption(data.get("generated_at", ""))

total = payload.get("today_total_spend")
st.metric("総額", f"¥{total:,.0f}" if isinstance(total,(int,float)) else "-")

alloc = pd.DataFrame(payload.get("allocations", []))
st.subheader("媒体別配分")
if not alloc.empty:
    if "share" in alloc:  alloc["share(%)"] = (alloc["share"]*100).round(1)
    if "amount" in alloc: alloc["amount(¥)"] = alloc["amount"].round(0)
    st.dataframe(alloc, use_container_width=True)
else:
    st.info("媒体別配分データがありません")

report = payload.get("report", {})
if report:
    st.subheader("要約")
    st.write(report.get("executive_summary") or "")

csv_url = st.secrets.get("CSV_URL")  # 任意（CSV公開する場合のみ）
if csv_url:
    try:
        csv_bytes = requests.get(csv_url, timeout=15).content
        st.download_button("compiled_wide.csv をダウンロード", csv_bytes, "compiled_wide.csv", "text/csv")
    except Exception as e:
        st.warning(f"CSV取得に失敗: {e}")
