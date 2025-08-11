import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

import os
from dotenv import load_dotenv
import tempfile
import boto3
from openai import OpenAI

# --- ページ設定 ---
st.set_page_config(page_title="アニメドラえもん あらすじ検索", page_icon="📚")

st.title("アニメドラえもん あらすじ検索")
st.markdown("アニメドラえもんで放送されたおはなしを, 登場人物や覚えている話の内容から検索できます")
st.markdown("データソース: [ドラえもん おはなしリスト](https://www.tv-asahi.co.jp/doraemon/story/bk/)")

# --- FAISS 読み込み ---
@st.cache_resource
def load_vectorstore():
    load_dotenv()
    OPENAPI_API_KEY = os.getenv("chatgpt_secret")
    S3_AWS_REGION = os.getenv("S3_AWS_REGION")
    S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME_ANIME")
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
query = st.text_area("🔍 おはなしのあらすじや特徴を入力してください", height=100)

# --- 検索ボタン ---
if st.button("検索"):
    if not query.strip():
        st.warning("入力が空です。あらすじを入力してください。")
    else:
        with st.spinner("検索中..."):
            # OpenAI APIでベクトル検索向けにクエリを補正（v1系対応）
            client = OpenAI(api_key=os.getenv("chatgpt_secret"))
            prompt = (
                "あなたは日本語の検索クエリをベクトル検索で最大限ヒットしやすいキーワード列に変換するAIです。"
                "次のユーザー入力を、不要な語を省き、検索意図が伝わるような簡潔なキーワード列に変換してください。\n"
                f"ユーザー入力: {query}\n"
                "ベクトル検索用キーワード列:"
            )
            response = client.chat.completions.create(
                model="gpt-5-mini",
                messages=[
                    {"role": "system", "content": "あなたは日本語の検索クエリをベクトル検索向けに最適化するAIです。"},
                    {"role": "user", "content": prompt}
                ],
                max_completion_tokens=64,
                # temperature=0.1,
            )
            refined_query = response.choices[0].message.content.strip()
            results = vectorstore.similarity_search_with_score(refined_query, k=3)

        st.subheader("🔎 類似エピソード")
        for i, (doc, score) in enumerate(results, 1):
            meta = doc.metadata
            title = meta.get("title", "タイトル不明")
            broadcasting_date = meta.get("broadcasting_date", "不明")
            story_index = meta.get("index", "不明")
            summary = doc.page_content.strip()

            with st.expander(f"{i}. {title}（話数: {story_index}, 類似度スコア: {score:.4f}）"):
                st.markdown("**放送日**: " + broadcasting_date)
                st.markdown("**要約**:")
                st.write(summary)
