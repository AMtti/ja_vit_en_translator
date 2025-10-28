# ==========================================================
# Streamlit ローカル翻訳アプリ（日本語 → ベトナム語／英語）
# モデル固定: facebook/m2m100_418M
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
# Streamlitキャッシュを起動時にクリア（Cloudのみ）
# ----------------------------------------------------------

def is_streamlit_cloud() -> bool:
    home = os.path.expanduser("~")
    return home.startswith("/app")
st.write(is_streamlit_cloud())
if is_streamlit_cloud():
    try:
        st.cache_data.clear()
        st.cache_resource.clear()
        st.info("☁️ Streamlit Cloud環境：キャッシュを初期化しました。")
    except Exception as e:
        st.warning(f"キャッシュ初期化時に問題が発生しました: {e}")
else:
    st.caption("💻 ローカル環境：キャッシュを保持して高速起動します。")

# PyTorch + Streamlit の警告を減らす
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

# モデルは固定
MODEL_NAME = "facebook/m2m100_418M"
SRC_LANG = "ja"
TGT_LANG = "vi" if "ベトナム" in target_lang else "en"

# ----------------------------------------------------------
# モデルの読み込み（キャッシュ付き）
# ----------------------------------------------------------
@st.cache_resource(show_spinner=True)
def load_model(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )

       
    model.eval()
    if torch.cuda.is_available():
        model.to("cuda")
    return tokenizer, model


tokenizer, model = load_model(MODEL_NAME)

# ----------------------------------------------------------
# UI
# ----------------------------------------------------------
st.title("日本語 → ベトナム語／英語 翻訳（ローカル）")
st.caption("モデル: facebook/m2m100_418M（オフライン可）")

ja_text = st.text_area("日本語テキストを入力", height=180, placeholder="例: この製品は炭素鋼SS400を使用しています。")
max_new_tokens = st.slider("最大出力トークン数", 32, 512, 256, step=32)

translate_btn = st.button("翻訳する")

# ----------------------------------------------------------
# 翻訳処理
# ----------------------------------------------------------
def translate(text: str) -> str:
    tokenizer.src_lang = SRC_LANG
    enc = tokenizer([text], return_tensors="pt", padding=True, truncation=True)
    if torch.cuda.is_available():
        enc = {k: v.to("cuda") for k, v in enc.items()}

    forced_bos_id = tokenizer.get_lang_id(TGT_LANG)

    gen = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        forced_bos_token_id=forced_bos_id
    )
    out = tokenizer.batch_decode(gen, skip_special_tokens=True)[0]
    return out


# ----------------------------------------------------------
# 出力表示
# ----------------------------------------------------------
if translate_btn and ja_text.strip():
    with st.spinner("翻訳中..."):
        result = translate(ja_text.strip())

    st.subheader(f"{target_lang} の翻訳結果")
    st.text_area("出力", value=result, height=180)

# ----------------------------------------------------------
# 注意書き
# ----------------------------------------------------------
st.markdown("---")
st.caption("""
💡 初回のみモデル (~1.2GB) をダウンロードします。
以降はオフラインで利用可能です。
""")
