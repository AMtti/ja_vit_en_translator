# ==========================================================
# Streamlit ローカル翻訳アプリ（日本語 → ベトナム語／英語）
# モデル固定: facebook/m2m100_418M（ローカル専用）
# ==========================================================

import os
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ----------------------------------------------------------
# 起動設定
# ----------------------------------------------------------
st.set_page_config(page_title="JA Translator (Offline)", layout="centered")

# ----------------------------------------------------------
# ローカルモデルパス（固定）
# ----------------------------------------------------------
MODEL_DIR = r".\models\facebook\m2m100_418M"  # ← ローカルモデルの絶対／相対パス
SRC_LANG = "ja"

# モデルフォルダ存在チェック
if not os.path.exists(MODEL_DIR):
    st.error(f"モデルディレクトリが見つかりません: {MODEL_DIR}")
    st.stop()

# ----------------------------------------------------------
# 完全オフライン設定
# ----------------------------------------------------------
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

# ----------------------------------------------------------
# 言語設定UI
# ----------------------------------------------------------
st.sidebar.title("設定")
target_lang = st.sidebar.selectbox(
    "翻訳先を選択してください",
    ["ベトナム語 (vi)", "英語 (en)"],
    index=0
)
TGT_LANG = "vi" if "ベトナム" in target_lang else "en"

# ----------------------------------------------------------
# モデルの読み込み（ローカル限定・キャッシュ付き）
# ----------------------------------------------------------
@st.cache_resource(show_spinner=True)
def load_model_local(model_dir: str):
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_dir,
        local_files_only=True,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True
    )
    model.eval()
    if torch.cuda.is_available():
        model.to("cuda")
    return tokenizer, model


tokenizer, model = load_model_local(MODEL_DIR)

# ----------------------------------------------------------
# UI
# ----------------------------------------------------------
st.title("日本語 → ベトナム語／英語 翻訳（ローカル）")
st.caption("モデル: facebook/m2m100_418M（完全オフライン）")

ja_text = st.text_area(
    "日本語テキストを入力",
    height=180,
    placeholder="例: この製品は炭素鋼SS400を使用しています。"
)
max_new_tokens = st.slider("最大出力トークン数", 32, 512, 256, step=32)

if st.button("翻訳する"):
    if ja_text.strip():
        with st.spinner("翻訳中..."):
            # 翻訳処理
            tokenizer.src_lang = SRC_LANG
            enc = tokenizer([ja_text.strip()], return_tensors="pt", padding=True, truncation=True)
            if torch.cuda.is_available():
                enc = {k: v.to("cuda") for k, v in enc.items()}

            forced_bos_id = tokenizer.get_lang_id(TGT_LANG)
            with torch.inference_mode():
                gen = model.generate(
                    **enc,
                    max_new_tokens=max_new_tokens,
                    forced_bos_token_id=forced_bos_id
                )
            result = tokenizer.batch_decode(gen, skip_special_tokens=True)[0]

        st.subheader(f"{target_lang} の翻訳結果")
        st.text_area("出力", value=result, height=180)
    else:
        st.warning("翻訳するテキストを入力してください。")

# ----------------------------------------------------------
# 注意書き
# ----------------------------------------------------------
st.markdown("---")
st.caption("""
💡 このアプリはローカル保存済みモデルのみを使用します（完全オフライン動作）。
モデルフォルダ: .\\models\\facebook\\m2m100_418M
""")
