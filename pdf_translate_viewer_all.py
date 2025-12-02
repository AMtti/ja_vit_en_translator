import os
import sys
from pathlib import Path

import pdfplumber
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QTextEdit,
    QFileDialog, QMessageBox, QComboBox, QProgressDialog
)
from PyQt6.QtCore import Qt

# ----------------------------------------------------------
# ローカルモデル設定
# ----------------------------------------------------------
MODEL_DIR = r".\models\facebook\m2m100_418M"
SRC_LANG = "ja"  # 入力は日本語


class PdfTextExtractorApp(QMainWindow):
    def __init__(self) -> None:
        super().__init__()

        self.setWindowTitle("PDF テキスト抽出＋翻訳ツール（PyQt6 + M2M100）")
        self.resize(1000, 800)

        self.pdf_path: Path | None = None
        self.page_count: int = 0

        # 翻訳モデル関連
        self.tokenizer = None
        self.model = None
        self.device = torch.device("cpu")
        self.translation_ready: bool = False

        self._setup_ui()
        self._load_translation_model()

    # ----------------------------------------
    # UI構築
    # ----------------------------------------
    def _setup_ui(self) -> None:
        central = QWidget(self)
        self.setCentralWidget(central)

        main_layout = QVBoxLayout(central)

        # ---- 上：ファイル選択エリア ----
        file_layout = QHBoxLayout()
        self.btn_open = QPushButton("PDFを開く...")
        self.btn_open.clicked.connect(self.open_pdf)

        self.lbl_file = QLabel("PDFファイル: （未選択）")
        self.lbl_file.setWordWrap(True)

        file_layout.addWidget(self.btn_open)
        file_layout.addWidget(self.lbl_file, stretch=1)

        # ---- 中：ページ選択エリア（プルダウン）----
        page_layout = QHBoxLayout()
        self.lbl_page = QLabel("ページ:")

        self.combo_page = QComboBox()
        self.combo_page.setEnabled(False)
        self.combo_page.currentIndexChanged.connect(self.on_page_changed_combo)

        self.lbl_page_total = QLabel("/ 0 ページ")

        self.btn_save_current = QPushButton("このページのテキストを保存")
        self.btn_save_current.setEnabled(False)
        self.btn_save_current.clicked.connect(self.save_current_page_text)

        self.btn_save_all = QPushButton("全ページをまとめて保存（原文）")
        self.btn_save_all.setEnabled(False)
        self.btn_save_all.clicked.connect(self.save_all_pages_text)

        page_layout.addWidget(self.lbl_page)
        page_layout.addWidget(self.combo_page)
        page_layout.addWidget(self.lbl_page_total)
        page_layout.addStretch()
        page_layout.addWidget(self.btn_save_current)
        page_layout.addWidget(self.btn_save_all)

        # ---- 元テキスト表示エリア ----
        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(False)
        self.text_edit.setPlaceholderText(
            "ここにPDFから抽出した日本語テキストが表示されます。\n"
            "PDFを開かずに、ここに直接日本語を書いて翻訳することもできます。"
        )

        font = self.text_edit.font()
        font.setFamily("Consolas")
        self.text_edit.setFont(font)

        # ---- 翻訳設定エリア ----
        trans_ctrl_layout = QHBoxLayout()
        self.lbl_target_lang = QLabel("翻訳先:")
        self.combo_lang = QComboBox()
        # 表示名, データ（言語コード）
        self.combo_lang.addItem("ベトナム語", "vi")
        self.combo_lang.addItem("英語", "en")

        self.btn_translate = QPushButton("入力欄のテキストを翻訳（日本語→選択言語）")
        self.btn_translate.setEnabled(False)  # モデル読み込み後に有効化
        self.btn_translate.clicked.connect(self.translate_current_page)

        self.btn_translate_all = QPushButton("全ページを翻訳して保存")
        self.btn_translate_all.setEnabled(False)
        self.btn_translate_all.clicked.connect(self.translate_and_save_all_pages)

        self.btn_save_translated = QPushButton("翻訳結果を保存（このページ）")
        self.btn_save_translated.setEnabled(False)
        self.btn_save_translated.clicked.connect(self.save_translated_text)

        trans_ctrl_layout.addWidget(self.lbl_target_lang)
        trans_ctrl_layout.addWidget(self.combo_lang)
        trans_ctrl_layout.addStretch()
        trans_ctrl_layout.addWidget(self.btn_translate)
        trans_ctrl_layout.addWidget(self.btn_translate_all)
        trans_ctrl_layout.addWidget(self.btn_save_translated)

        # ---- 翻訳結果表示エリア ----
        self.text_translated = QTextEdit()
        self.text_translated.setReadOnly(False)
        self.text_translated.setPlaceholderText("ここに翻訳結果が表示されます。")

        font2 = self.text_translated.font()
        font2.setFamily("Consolas")
        self.text_translated.setFont(font2)

        # レイアウトをメインに追加
        main_layout.addLayout(file_layout)
        main_layout.addLayout(page_layout)
        main_layout.addWidget(QLabel("📘 原文（日本語・編集／直接入力可）"))
        main_layout.addWidget(self.text_edit, stretch=1)
        main_layout.addLayout(trans_ctrl_layout)
        main_layout.addWidget(QLabel("🌏 翻訳結果"))
        main_layout.addWidget(self.text_translated, stretch=1)

    # ----------------------------------------
    # 翻訳モデルの読み込み（ローカル）
    # ----------------------------------------
    def _load_translation_model(self) -> None:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

        try:
            if not Path(MODEL_DIR).exists():
                raise FileNotFoundError(f"モデルディレクトリが見つかりません: {MODEL_DIR}")

            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)

            self.tokenizer.src_lang = SRC_LANG

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            self.model.eval()

            self.translation_ready = True

            # モデル準備OKなら、PDFなしでも翻訳ボタンを有効にする
            if hasattr(self, "btn_translate"):
                self.btn_translate.setEnabled(True)

        except Exception as e:
            QMessageBox.warning(
                self,
                "翻訳モデルエラー",
                f"翻訳モデルの読み込みに失敗しました:\n{e}\n翻訳機能は無効化されます。"
            )
            self.translation_ready = False

    # ----------------------------------------
    # PDFを開く
    # ----------------------------------------
    def open_pdf(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "PDFファイルを選択",
            "",
            "PDF Files (*.pdf);;All Files (*)"
        )
        if not file_path:
            return

        path = Path(file_path)

        if not path.exists():
            QMessageBox.warning(self, "エラー", "選択したファイルが存在しません。")
            return

        try:
            with pdfplumber.open(path) as pdf:
                page_count = len(pdf.pages)
        except Exception as e:
            QMessageBox.critical(self, "エラー", f"PDFを開けませんでした:\n{e}")
            return

        self.pdf_path = path
        self.page_count = page_count

        self.lbl_file.setText(f"PDFファイル: {str(path)}")

        # ページ番号をコンボボックスに設定
        self.combo_page.setEnabled(True)
        self.combo_page.clear()
        for i in range(1, page_count + 1):
            self.combo_page.addItem(str(i))
        self.combo_page.setCurrentIndex(0)

        self.lbl_page_total.setText(f"/ {page_count} ページ")

        self.btn_save_current.setEnabled(True)
        self.btn_save_all.setEnabled(True)

        # 全ページ翻訳ボタンは、モデルがロードできていれば有効化
        self.btn_translate_all.setEnabled(self.translation_ready)

        # 1ページ目を表示
        self.load_page_text(1)

    # ----------------------------------------
    # ページ選択（コンボボックス）の変更時
    # ----------------------------------------
    def on_page_changed_combo(self, index: int) -> None:
        """ComboBox の index は 0 始まり → ページ番号は index+1"""
        if self.pdf_path is None:
            return
        if index < 0:
            return
        page_number = index + 1
        self.load_page_text(page_number)

    # ----------------------------------------
    # 指定ページのテキストを読み込み
    # ----------------------------------------
    def load_page_text(self, page_number: int) -> None:
        """1始まりの page_number で指定"""
        if self.pdf_path is None:
            return

        try:
            with pdfplumber.open(self.pdf_path) as pdf:
                index = page_number - 1  # pdfplumber は 0 始まり
                if index < 0 or index >= len(pdf.pages):
                    raise IndexError("ページ番号が範囲外です。")

                page = pdf.pages[index]
                text = page.extract_text() or ""

        except Exception as e:
            QMessageBox.critical(self, "エラー", f"ページの読み込みに失敗しました:\n{e}")
            return

        if not text.strip():
            text = "[このページからテキストを抽出できませんでした。画像のみのページの可能性があります。]"

        self.text_edit.setPlainText(text)
        # ページを切り替えたら翻訳結果はいったんクリア
        self.text_translated.clear()
        self.btn_save_translated.setEnabled(False)

    # ----------------------------------------
    # このページのテキストを保存（日本語）
    # ----------------------------------------
    def save_current_page_text(self) -> None:
        text = self.text_edit.toPlainText()
        if not text.strip():
            QMessageBox.information(self, "情報", "保存するテキストがありません。")
            return

        # 現在のページ番号（1始まり）
        current_page = self.combo_page.currentIndex() + 1 if self.combo_page.count() > 0 else 1

        default_name = "page_text.txt"
        if self.pdf_path:
            default_name = f"{self.pdf_path.stem}_page{current_page}_ja.txt"

        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "テキストファイルとして保存（日本語）",
            default_name,
            "Text Files (*.txt);;All Files (*)"
        )
        if not save_path:
            return

        try:
            Path(save_path).write_text(text, encoding="utf-8")
        except Exception as e:
            QMessageBox.critical(self, "エラー", f"ファイル保存に失敗しました:\n{e}")
            return

        QMessageBox.information(self, "完了", "テキストを保存しました。")

    # ----------------------------------------
    # 全ページのテキストをまとめて保存（日本語）
    # ----------------------------------------
    def save_all_pages_text(self) -> None:
        if self.pdf_path is None:
            QMessageBox.warning(self, "エラー", "PDFが選択されていません。")
            return

        default_name = "all_pages_ja.txt"
        if self.pdf_path:
            default_name = f"{self.pdf_path.stem}_all_pages_ja.txt"

        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "全ページのテキストを保存（日本語）",
            default_name,
            "Text Files (*.txt);;All Files (*)"
        )
        if not save_path:
            return

        try:
            all_text_parts: list[str] = []
            with pdfplumber.open(self.pdf_path) as pdf:
                total = len(pdf.pages)
                for i, page in enumerate(pdf.pages, start=1):
                    text = page.extract_text() or ""
                    header = f"===== ページ {i} / {total} =====\n"
                    all_text_parts.append(header + text + "\n\n")

            result_text = "".join(all_text_parts)
            Path(save_path).write_text(result_text, encoding="utf-8")

        except Exception as e:
            QMessageBox.critical(self, "エラー", f"全ページ保存中にエラーが発生しました:\n{e}")
            return

        QMessageBox.information(self, "完了", "全ページの日本語テキストを保存しました。")

    # ----------------------------------------
    # 実際の翻訳処理（日本語 → tgt_lang_code）
    # 1行ごとに翻訳して、改行位置を揃える
    # progress_dialog が渡された場合は行ごとに進捗更新
    # ----------------------------------------
    def _translate_text(self, text: str, tgt_lang_code: str,
                        progress_dialog: QProgressDialog | None = None) -> str:
        if not self.translation_ready or self.tokenizer is None or self.model is None:
            raise RuntimeError("翻訳モデルが初期化されていません。")

        lines = text.splitlines()
        forced_bos_token_id = self.tokenizer.get_lang_id(tgt_lang_code)
        translated_lines: list[str] = []

        current_value = progress_dialog.value() if progress_dialog is not None else 0

        for line in lines:
            # 完全な空行はそのまま
            if not line.strip():
                translated_lines.append("")
                if progress_dialog is not None:
                    current_value += 1
                    progress_dialog.setValue(current_value)
                    QApplication.processEvents()
                    if progress_dialog.wasCanceled():
                        raise RuntimeError("ユーザーがキャンセルしました。")
                continue

            encoded = self.tokenizer(
                line,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            encoded = {k: v.to(self.device) for k, v in encoded.items()}

            with torch.no_grad():
                generated = self.model.generate(
                    **encoded,
                    forced_bos_token_id=forced_bos_token_id,
                    max_length=512,
                    num_beams=4,
                )

            out = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
            translated_lines.append(out[0])

            if progress_dialog is not None:
                current_value += 1
                progress_dialog.setValue(current_value)
                QApplication.processEvents()
                if progress_dialog.wasCanceled():
                    raise RuntimeError("ユーザーがキャンセルしました。")

        return "\n".join(translated_lines)

    # ----------------------------------------
    # 現在のテキスト（入力欄の中身）を翻訳（プログレスバー付き）
    # ----------------------------------------
    def translate_current_page(self) -> None:
        if not self.translation_ready:
            QMessageBox.warning(self, "翻訳エラー", "翻訳モデルが読み込まれていません。")
            return

        src_text = self.text_edit.toPlainText()
        if not src_text.strip():
            QMessageBox.information(self, "情報", "翻訳するテキストがありません。")
            return

        tgt_lang_code = self.combo_lang.currentData()  # "vi" or "en"

        lines = src_text.splitlines()
        total_steps = len(lines) if lines else 1

        progress = QProgressDialog("テキストを翻訳中です…", "キャンセル", 0, total_steps, self)
        progress.setWindowTitle("翻訳中")
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)

        try:
            translated = self._translate_text(src_text, tgt_lang_code, progress_dialog=progress)
        except RuntimeError as e:
            if "キャンセル" in str(e):
                QMessageBox.information(self, "中断", "翻訳をキャンセルしました。")
            else:
                QMessageBox.critical(self, "翻訳エラー", f"翻訳中にエラーが発生しました:\n{e}")
            return
        except Exception as e:
            QMessageBox.critical(self, "翻訳エラー", f"翻訳中にエラーが発生しました:\n{e}")
            return
        finally:
            progress.close()

        self.text_translated.setPlainText(translated)
        self.btn_save_translated.setEnabled(True)

    # ----------------------------------------
    # 全ページを翻訳して保存（プログレスバー付き）
    # ----------------------------------------
    def translate_and_save_all_pages(self) -> None:
        if self.pdf_path is None:
            QMessageBox.warning(self, "エラー", "PDFが選択されていません。")
            return
        if not self.translation_ready:
            QMessageBox.warning(self, "翻訳エラー", "翻訳モデルが読み込まれていません。")
            return

        tgt_lang_code = self.combo_lang.currentData()
        tgt_lang_label = "vi" if tgt_lang_code == "vi" else "en"

        default_name = "all_pages_translated.txt"
        if self.pdf_path:
            default_name = f"{self.pdf_path.stem}_all_pages_{tgt_lang_label}.txt"

        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "全ページの翻訳結果を保存",
            default_name,
            "Text Files (*.txt);;All Files (*)"
        )
        if not save_path:
            return

        # ページ数をステップ数とする
        try:
            with pdfplumber.open(self.pdf_path) as pdf:
                total_pages = len(pdf.pages)
        except Exception as e:
            QMessageBox.critical(self, "エラー", f"PDFオープンに失敗しました:\n{e}")
            return

        progress = QProgressDialog("全ページを翻訳中です…", "キャンセル", 0, total_pages, self)
        progress.setWindowTitle("翻訳中")
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)

        try:
            all_text_parts: list[str] = []
            with pdfplumber.open(self.pdf_path) as pdf:
                total = len(pdf.pages)
                for i, page in enumerate(pdf.pages, start=1):
                    if progress.wasCanceled():
                        raise RuntimeError("ユーザーがキャンセルしました。")

                    src_text = page.extract_text() or ""
                    header = f"===== ページ {i} / {total} =====\n"

                    if src_text.strip():
                        translated = self._translate_text(src_text, tgt_lang_code)
                    else:
                        translated = "[このページには翻訳対象のテキストがありません。]"

                    all_text_parts.append(header + translated + "\n\n")

                    progress.setValue(i)
                    QApplication.processEvents()

            result_text = "".join(all_text_parts)
            Path(save_path).write_text(result_text, encoding="utf-8")

        except RuntimeError as e:
            if "キャンセル" in str(e):
                QMessageBox.information(self, "中断", "翻訳をキャンセルしました。")
            else:
                QMessageBox.critical(self, "翻訳エラー", f"全ページ翻訳中にエラーが発生しました:\n{e}")
            return
        except Exception as e:
            QMessageBox.critical(self, "翻訳エラー", f"全ページ翻訳中にエラーが発生しました:\n{e}")
            return
        finally:
            progress.close()

        QMessageBox.information(self, "完了", "全ページの翻訳結果を保存しました。")

    # ----------------------------------------
    # 翻訳結果を保存（現在ページ）
    # ----------------------------------------
    def save_translated_text(self) -> None:
        text = self.text_translated.toPlainText()
        if not text.strip():
            QMessageBox.information(self, "情報", "保存する翻訳テキストがありません。")
            return

        current_page = self.combo_page.currentIndex() + 1 if self.combo_page.count() > 0 else 1
        tgt_lang_code = self.combo_lang.currentData()
        tgt_lang_label = "vi" if tgt_lang_code == "vi" else "en"

        default_name = "page_translated.txt"
        if self.pdf_path:
            default_name = f"{self.pdf_path.stem}_page{current_page}_{tgt_lang_label}.txt"

        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "翻訳結果をテキストファイルとして保存（このページ）",
            default_name,
            "Text Files (*.txt);;All Files (*)"
        )
        if not save_path:
            return

        try:
            Path(save_path).write_text(text, encoding="utf-8")
        except Exception as e:
            QMessageBox.critical(self, "エラー", f"翻訳テキストの保存に失敗しました:\n{e}")
            return

        QMessageBox.information(self, "完了", "翻訳結果を保存しました。")


def main() -> None:
    app = QApplication(sys.argv)
    window = PdfTextExtractorApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
