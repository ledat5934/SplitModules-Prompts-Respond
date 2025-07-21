from __future__ import annotations

from pathlib import Path
from typing import List

import nbformat as nbf # Vẫn cần import này nếu không muốn xóa hoàn toàn nó, hoặc có thể xóa nếu không sử dụng gì nữa.

# Thêm các import mới
import google.generativeai as genai
import os
from dotenv import load_dotenv # Đảm bảo bạn đã cài đặt: pip install python-dotenv

# ----------------------------------------------------------------------
# Optional project utilities.  Fall back to lightweight stubs if missing.
# ----------------------------------------------------------------------
try:
    from src.utils.logger import get_logger  # type: ignore
except Exception:  # pragma: no cover
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    get_logger = logging.getLogger  # type: ignore

logger = get_logger(__name__)

# --- Hàm loại bỏ ký tự code block ---
def remove_code_block_markers(text: str) -> str:
    """
    Removes Markdown code block markers (```python and ```) from a string.

    Args:
        text (str): The input string potentially containing code block markers.

    Returns:
        str: The string with code block markers removed.
    """
    # Loại bỏ '```python' trước để tránh làm hỏng '```' đơn thuần
    text = text.replace("```python", "")
    # Loại bỏ '```'
    text = text.replace("```", "")
    return text
# --- Kết thúc hàm loại bỏ ký tự code block ---


class CodeAssembler:
    """Combine stage scripts into a single, coherent Python module + notebook."""

    # ------------------------------------------------------------------
    # Internal – optional AI refactor
    # ------------------------------------------------------------------
    @staticmethod
    def _clean_with_ai(code: str) -> str:
        """
        Send *code* to Gemini (if available) to de-duplicate imports,
        ensure a single `main()` + `if __name__ == '__main__':`, etc.

        If Gemini SDK or keys are absent, returns *code* unchanged.
        """
        # Tải biến môi trường từ tệp .env
        load_dotenv()
        GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

        if not GEMINI_API_KEY:
            logger.warning("GEMINI_API_KEY not found in environment variables or .env file → skipping AI cleaning.")
            return code

        # Cấu hình API key cho thư viện google.generativeai
        genai.configure(api_key=GEMINI_API_KEY)

        try:
            # Khởi tạo model Gemini 2.0 Flash
            model = genai.GenerativeModel('gemini-2.0-flash')

        except Exception as e:
            logger.error(f"Failed to load Gemini model 'gemini-2.0-flash': {e} → skipping AI cleaning.")
            return code

        system_prompt = (
            "You are an expert Python engineer. You will receive a Python script "
            "produced by concatenating several files; it might contain duplicate "
            "imports, multiple `if __name__ == '__main__':` blocks, and redundant "
            "helpers. Refactor into ONE clean, executable script:\n"
            "1. Merge imports at top (remove duplicates).\n"
            "2. Provide exactly ONE main entry point; move top-level logic into "
            "a `main()` function and call it.\n"
            "3. Remove duplicate functions/classes (keep first version).\n"
            "4. Do NOT alter behaviour.\n"
            "5. The output code should run in the train set, not the sample or dummy data.\n"
            "Output ONLY plain Python source (no markdown)."
        )

        try:
            response = model.generate_content(
                contents=[
                    {"role": "user", "parts": [system_prompt]},
                    {"role": "user", "parts": [f"Here is the script:\n\n{code}"]},
                ],
            )
            
            cleaned_raw_text = response.text
            cleaned_code = remove_code_block_markers(cleaned_raw_text)

            logger.info("AI cleaning finished.")
            return cleaned_code
        except genai.types.BlockedPromptException as e:
            logger.error(f"Gemini cleaning failed due to safety settings: {e}. Using un-cleaned code.", exc_info=True)
            return code
        except Exception as exc:  # pragma: no cover
            logger.error("Gemini cleaning failed (%s). Using un-cleaned code.", exc, exc_info=True)
            return code

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def assemble(
        self,
        project_name: str,
        stage_files: List[Path],
        output_root: Path = Path("generated_code"),
    ) -> None:
        """Concatenate *stage_files* and write outputs under *output_root*."""
        if not stage_files:
            raise ValueError("No stage files provided to assemble")

        output_root.mkdir(parents=True, exist_ok=True)
        py_file = output_root / f"{project_name}.py"
        # nb_file = output_root / f"{project_name}.ipynb" # Dòng này đã được comment/xóa

        logger.info(
            "Assembling %d files → %s", # Đã bỏ '%s' thứ hai và đối số nb_file
            len(stage_files), py_file # Đã bỏ nb_file khỏi log
        )

        # -- read & concatenate ------------------------------------------------
        snippets = [p.read_text(encoding="utf-8") for p in stage_files]
        combined_code = "\n\n".join(snippets)

        # -- optional AI clean -------------------------------------------------
        cleaned_code = self._clean_with_ai(combined_code)
        py_file.write_text(cleaned_code, encoding="utf-8")
        logger.info("Wrote cleaned script: %s", py_file)

        # -- notebook with original cells -------------------------------------
        # Các dòng này đã được comment/xóa để không tạo notebook
        # nb = nbf.v4.new_notebook()
        # nb.cells.extend(nbf.v4.new_code_cell(s) for s in snippets)
        # nbf.write(nb, str(nb_file))
        # logger.info("Wrote notebook:       %s", nb_file)


# ----------------------------------------------------------------------
# CLI wrapper (convenience)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Combine multiple stage scripts into one .py and .ipynb."
    )
    parser.add_argument("project_name", help="Base name for output files (without extension)")
    parser.add_argument("scripts", nargs="+", type=Path, help="Stage *.py files in desired order")
    parser.add_argument("--out", default="generated_code", help="Output directory")
    args = parser.parse_args()

    assembler = CodeAssembler()
    assembler.assemble(args.project_name, args.scripts, Path(args.out))