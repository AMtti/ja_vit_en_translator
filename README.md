

# 🇯🇵➡️🇻🇳🇬🇧 JA → VI / EN Translator (Streamlit App)

日本語の文章を **ベトナム語** または **英語** に翻訳できるシンプルな Streamlit アプリです。
ローカルでも、[Streamlit Cloud](https://share.streamlit.io) 上でも動作します。

---

## 🌐 アプリ概要

このアプリは Meta 社の多言語モデル
[`facebook/m2m100_418M`](https://huggingface.co/facebook/m2m100_418M)
を使用して、日本語から複数言語への高品質な機械翻訳を行います。

* **入力言語:** 日本語（固定）
* **出力言語:** ベトナム語 🇻🇳 または 英語 🇬🇧
* **利用モデル:** facebook/m2m100_418M（約1.2GB）

---

## 🧠 使用技術

| 種類       | 技術                                                              |
| -------- | --------------------------------------------------------------- |
| フレームワーク  | [Streamlit](https://streamlit.io/)                              |
| モデルライブラリ | [Transformers](https://huggingface.co/docs/transformers)        |
| モデル提供元   | [Meta AI (M2M100)](https://huggingface.co/facebook/m2m100_418M) |
| 推論エンジン   | [PyTorch](https://pytorch.org/)                                 |
| 文字分割     | [SentencePiece](https://github.com/google/sentencepiece)        |

---

## ⚙️ セットアップ方法（ローカル実行）

1. Python をインストール（3.10〜3.12 推奨）
2. このリポジトリをクローン

   ```bash
   git clone https://github.com/AMtti/ja_vit_en_translator.git
   cd ja_vit_en_translator
   ```
3. 必要ライブラリをインストール

   ```bash
   pip install -r requirements.txt
   ```
4. アプリを起動

   ```bash
   streamlit run app_ja_translator.py
   ```
5. ブラウザで表示

   ```
   http://localhost:8501
   ```

---

## ☁️ Streamlit Cloud での公開方法

1. GitHub リポジトリを作成（Public）
2. `app_ja_translator.py` と `requirements.txt` をアップロード
3. [Streamlit Cloud](https://share.streamlit.io) にアクセス
4. 「New app」→ リポジトリを選択
5. メインファイルパスに以下を指定して「Deploy」

   ```
   app_ja_translator.py
   ```

📌 初回起動時のみモデル（約1.2GB）のダウンロードが行われます。
以降はキャッシュを利用して高速に動作します。

---

## 💡 サンプル入力

```
この製品は炭素鋼SS400を使用しています。
```

**出力（ベトナム語）**

```
Sản phẩm này sử dụng thép cacbon SS400.
```

**出力（英語）**

```
This product uses carbon steel SS400.
```

---

## 🧩 フォルダ構成

```
ja_vit_en_translator/
├── app_ja_translator.py   # Streamlitアプリ本体
├── requirements.txt       # 依存ライブラリ
└── README.md              # この説明ファイル
```

---

## 🧑‍💻 開発者メモ

* 最初の翻訳リクエスト時はモデルのロードに 2〜4 分ほどかかります。
* GPU 環境（例: Streamlit Cloud Premium / Colab / ローカル GPU）では高速に動作します。
* Hugging Face Hub のキャッシュ機構を利用して、2回目以降の翻訳は即時応答します。

---

## 📜 ライセンス

MIT License
© 2025 AMtti

---


