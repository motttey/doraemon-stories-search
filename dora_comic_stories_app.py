import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

import os
from dotenv import load_dotenv

# --- ページ設定 ---
st.set_page_config(page_title="ドラえもんのコミック検索", page_icon="📚")

st.title("ドラえもん あらすじから検索")

# --- FAISS 読み込み ---
@st.cache_resource
def load_vectorstore():
    # .envファイルから環境変数を読み込み
    load_dotenv()

    # openai用のAPIキーを取得
    OPENAPI_API_KEY = os.getenv("chatgpt_secret")

    embedding_model = OpenAIEmbeddings(openai_api_key=OPENAPI_API_KEY)
    return FAISS.load_local(
        "store/faiss_index",
        embeddings=embedding_model,
        allow_dangerous_deserialization=True,
    )

vectorstore = load_vectorstore()

# --- クエリ入力欄 ---
query = st.text_area("🔍 話のあらすじや特徴を入力してください", height=100)

# --- 検索ボタン ---
if st.button("検索"):
    if not query.strip():
        st.warning("入力が空です。あらすじを入力してください。")
    else:
        with st.spinner("検索中..."):
            results = vectorstore.similarity_search_with_score(query, k=3)

        st.subheader("🔎 類似エピソード")
        for i, (doc, score) in enumerate(results, 1):
            meta = doc.metadata
            title = meta.get("title", "タイトル不明")
            story_index = meta.get("story_index", "不明")
            volume = meta.get("volume", "不明")
            issue_info = meta.get("issue_info", "")
            summary = doc.page_content.strip()

            with st.expander(f"{i}. {title}（話数: {story_index}, 巻: {volume}, 類似度スコア: {score:.4f}）"):
                st.markdown("**掲載情報**: " + issue_info)
                st.markdown("**要約**:")
                st.write(summary)
