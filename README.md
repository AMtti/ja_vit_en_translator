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
📦 ja_vit_en_translator_m2m100_418M/
├─ m2m100_418M_streamlit.py                         ← Streamlit アプリ本体
├─ models_facebook_m2m100_418M.zip ← 圧縮済モデルファイル（展開して使用）
└─ README.md
```

展開後の構成：
```
📦  ja_vit_en_translator_m2m100_418M/
├─  m2m100_418M_streamlit.py
├─ models/
│   └─ facebook/
│       └─ m2m100_418M/
│           ├─ config.json
│           ├─ pytorch_model.bin
│           ├─ tokenizer.json
│           ├─ sentencepiece.bpe.model
│           └─ ...（その他ファイル）
└─ README.md
```

---

## 🛠️ セットアップ方法

### ① リポジトリをクローン
```bash
git clone https://github.com/AMtti/ja_vit_en_translator.git
cd ja_vit_en_translator_m2m100_418M
```

### ② モデルファイルを展開
zip を展開して、次の構成にしてください：

```bash
models_facebook_m2m100_418M.zip → models/facebook/m2m100_418M/
```

例（Windows PowerShell）：
```powershell
Expand-Archive .\models_facebook_m2m100_418M.zip -DestinationPath .\models\facebook\m2m100_418M
```

---

## 💻 実行方法

### 1. 仮想環境を作成（推奨）
```bash
python -m venv StreamlitApps
.\StreamlitApps\Scripts\activate
```

### 2. 依存ライブラリをインストール
```bash
pip install -r requirements.txt
```

もし `requirements.txt` がない場合は以下を実行：
```bash
pip install streamlit torch transformers
```

### 3. アプリを起動
```bash
streamlit run app.py
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

---

## 利用上の注意
このアプリは試作品です🙇
