import streamlit as st
import pandas as pd
from pandasai import SmartDataframe
from langchain_google_genai import ChatGoogleGenerativeAI

# 頁面基本設定
st.set_page_config(page_title="CSV 智能對話機械人", page_icon="🤖", layout="wide")

st.title("🤖 CSV 智能對話機械人 (Streamlit + PandasAI + Gemini)")
st.markdown("上傳你的遊戲數據或任何 CSV 檔案，即可直接用自然語言與數據進行對話、查詢數值或生成圖表！")

# 側邊欄：API 金鑰與設定
with st.sidebar:
    st.header("⚙️ 系統設定")
    google_api_key = st.text_input("請輸入 Google Gemini API Key", type="password")
    st.markdown("[取得 Google AI Studio API Key](https://aistudio.google.com/)")
    
    selected_model = st.selectbox(
        "選擇 Gemini 模型",
        ["gemini-2.5-flash", "gemini-1.5-pro", "gemini-1.5-flash"],
        index=0
    )
    st.markdown("---")
    st.info("提示：你可以在 Google AI Studio 免費申請 API Key。")

# 檔案上傳區塊
uploaded_file = st.file_uploader("上傳你的 CSV 檔案", type=["csv"])

if uploaded_file is not None:
    # 讀取 CSV 檔案
    try:
        df = pd.read_csv(uploaded_file)
        with st.expander("📊 點擊查看原始資料預覽 (前 5 行)"):
            st.dataframe(df.head())
    except Exception as e:
        st.error(f"讀取 CSV 失敗，請確認檔案格式是否正確：{e}")
        st.stop()

    if not google_api_key:
        st.warning("⚠️ 請先在左側欄位輸入你的 Google Gemini API Key 才能開始對話。")
    else:
        try:
            # 初始化 Gemini 模型與 PandasAI 智能 DataFrame
            llm = ChatGoogleGenerativeAI(model=selected_model, google_api_key=google_api_key)
            sdf = SmartDataframe(df, config={"llm": llm, "conversational": True})
            
            st.success("✅ 機械人已連線成功！請在下方輸入你想詢問的問題。")
            
            # 初始化對話歷史紀錄
            if "messages" not in st.session_state:
                st.session_state.messages = []

            # 顯示歷史對話
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            # 接收玩家輸入
            if prompt := st.chat_input("例如：『總共有多少筆資料？』或『幫我找出數值最高的前五名』"):
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

                with st.chat_message("assistant"):
                    with st.spinner("AI 正在分析數據中..."):
                        try:
                            # 透過 PandasAI 讓 Gemini 回應玩家提問
                            response = sdf.chat(prompt)
                            
                            # 如果 PandasAI 回傳的是圖表或文字，適當呈現
                            if response is not None:
                                st.markdown(response)
                                st.session_state.messages.append({"role": "assistant", "content": str(response)})
                            else:
                                msg = "已完成處理，但未產生文字回覆。"
                                st.markdown(msg)
                                st.session_state.messages.append({"role": "assistant", "content": msg})
                                
                        except Exception as e:
                            error_msg = f"❌ 分析時發生錯誤：{str(e)}"
                            st.error(error_msg)
                            st.session_state.messages.append({"role": "assistant", "content": error_msg})

        except Exception as e:
            st.error(f"初始化模型失敗，請檢查 API Key 或網路連線：{str(e)}")
else:
    st.info("👈 請先在上方上傳一個 CSV 檔案來啟動聊天機械人。")