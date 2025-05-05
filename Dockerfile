# ベースイメージ
FROM python:3.10-slim

# 必要なパッケージをインストール
RUN apt-get update && apt-get install -y curl

# 作業ディレクトリを設定
WORKDIR /app

# requirements.txtをコピーして依存関係をインストール
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# アプリのコードをコピー
COPY ./ ./

# ポートを公開
EXPOSE 8501

# Streamlit アプリを起動
CMD ["streamlit", "run", "dora_comic_stories_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
