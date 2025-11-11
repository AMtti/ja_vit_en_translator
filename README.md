# 🌐 JA Translator (Offline)
**日本語 → ベトナム語／英語 翻訳アプリ（完全ローカル動作）**

---

## 🧩 概要
このアプリは [facebook/m2m100_418M](https://huggingface.co/facebook/m2m100_418M) モデルを使用した  
**オフライン翻訳アプリ**です。  
インターネット接続なしで、ローカルの PC 上で日本語テキストを  
**ベトナム語**または**英語**に翻訳できます。

---


## 📁 ディレクトリ構成
```
📦 JA-Translator-Offline/
├─ m2m100_418M_streamlit.py                        ← Streamlit アプリ本体
├─ tools/
│   └─ download_model.py         ← モデルを自動ダウンロードするスクリプト
├─ models/                       ← モデル格納先（初回は空でもOK）
└─ README.md
```

---

## 💾 モデルの入手方法

GitHub にはモデルを含めていません（サイズが約1.2GBのため）。  
以下のスクリプトで自動ダウンロードしてください。

### 📜 `tools/download_model.py`
```python
# tools/download_model.py
from huggingface_hub import snapshot_download
from pathlib import Path

# 変更可: 保存先
TARGET_DIR = Path(__file__).resolve().parents[1] / "models" / "facebook" / "m2m100_418M"

if __name__ == "__main__":
    TARGET_DIR.parent.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id="facebook/m2m100_418M",
        local_dir=str(TARGET_DIR),
        local_dir_use_symlinks=False
    )
    print(f"Downloaded to: {TARGET_DIR}")
```

### 🪄 実行手順
```bash
cd tools
python download_model.py
```

完了後、次のフォルダが生成されます：
```
models/facebook/m2m100_418M/
```

これで `m2m100_418M_streamlit.py` がローカルモデルを自動認識して動作します。

---

## 🛠️ セットアップ方法

### ① リポジトリをクローン
```bash
git clone https://github.com/あなたのユーザー名/JA-Translator-Offline.git
cd JA-Translator-Offline
```

### ② 依存ライブラリをインストール
```bash
pip install -r requirements.txt
```

または手動で：
```bash
pip install streamlit torch transformers huggingface_hub
```

### ③ モデルをダウンロード
```bash
python tools/download_model.py
```

### ④ アプリを起動
```bash
streamlit run m2m100_418M_streamlit.py
```

ブラウザが自動的に開きます

---

## 🌍 使い方
1. 左サイドバーで翻訳先を選択（ベトナム語 or 英語）  
2. テキストエリアに日本語を入力  
3. 「翻訳する」ボタンをクリック  
4. 下部に翻訳結果が表示されます  

---

## ⚙️ 環境変数（内部設定）
アプリ内で自動的に次の環境変数が設定されます：
```python
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
```
→ Hugging Face Hub へのアクセスを遮断し、完全オフラインを保証します。

---

## ⚡ GPU 利用について
CUDA が有効な場合、自動的に `FP16` モードで GPU にモデルをロードします。  
CPU 環境でも動作しますが、翻訳速度はやや遅くなります。


---

## 📚 参考
- モデル: [facebook/m2m100_418M](https://huggingface.co/facebook/m2m100_418M)  
- Transformers: [https://github.com/huggingface/transformers](https://github.com/huggingface/transformers)  
- Streamlit: [https://streamlit.io](https://streamlit.io)  
- Hugging Face Hub CLI: [https://huggingface.co/docs/huggingface_hub](https://huggingface.co/docs/huggingface_hub)

---

## 利用上の注意！
このアプリは試作品です。
