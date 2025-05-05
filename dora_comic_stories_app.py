import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

import os
from dotenv import load_dotenv
import tempfile
import boto3

# --- ページ設定 ---
st.set_page_config(page_title="ドラえもんのコミック検索", page_icon="📚")

st.title("ドラえもん あらすじから検索")

# --- FAISS 読み込み ---
@st.cache_resource
def load_vectorstore():
    load_dotenv()
    OPENAPI_API_KEY = os.getenv("chatgpt_secret")
    S3_AWS_REGION = os.getenv("S3_AWS_REGION")
    S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
    IAM_ACCESS_KEY = os.getenv("IAM_ACCESS_KEY")
    IAM_SECRET_ACCESS_KEY = os.getenv("IAM_SECRET_ACCESS_KEY")

    embedding_model = OpenAIEmbeddings(openai_api_key=OPENAPI_API_KEY)

    # S3からFAISSファイル群を一時ディレクトリにダウンロード
    s3 = boto3.client(
        "s3",
        region_name=S3_AWS_REGION,
        aws_access_key_id=IAM_ACCESS_KEY,
        aws_secret_access_key=IAM_SECRET_ACCESS_KEY
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        files_to_download = ["index.faiss", "index.pkl"]  # 必要なファイル
        for file in files_to_download:
            s3.download_file(S3_BUCKET_NAME, file, os.path.join(tmpdir, file))

        # FAISSを読み込み
        vectorstore = FAISS.load_local(
            tmpdir,
            embeddings=embedding_model,
            allow_dangerous_deserialization=True,
        )
        return vectorstore


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
