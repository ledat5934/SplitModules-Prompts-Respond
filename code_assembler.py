"""
code_assembler.py
-----------------
Merge several stage scripts into one cleaned *.py* file and a companion
*.ipynb* notebook.

Usage (CLI)
-----------
python code_assembler.py my_project \
        generated_code/preprocessing_dataset_1.py \
        generated_code/modeling_dataset_1.py \
        --out combined

This creates:
    combined/
        my_project.py        # AI-cleaned unified script
        my_project.ipynb     # notebook with original cells
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import nbformat as nbf

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

try:
    from src.utils.env_config import load_config  # type: ignore
    from src.utils.gemini_client import GeminiClient  # type: ignore
except Exception:  # pragma: no cover
    load_config = None
    GeminiClient = None  # type: ignore

logger = get_logger(__name__)


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
        if not (load_config and GeminiClient):
            logger.warning("Gemini utilities unavailable → skipping AI cleaning.")
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
            cfg = load_config()
            client = GeminiClient(cfg)
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Here is the script:\n\n{code}"},
            ]
            cleaned = client.chat_completion(messages=messages)
            logger.info("AI cleaning finished.")
            return cleaned
        except Exception as exc:  # pragma: no cover
            logger.error("Gemini cleaning failed (%s). Using un-cleaned code.", exc)
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
        nb_file = output_root / f"{project_name}.ipynb"

        logger.info(
            "Assembling %d files → %s  &  %s",
            len(stage_files), py_file, nb_file
        )

        # -- read & concatenate ------------------------------------------------
        snippets = [p.read_text(encoding="utf-8") for p in stage_files]
        combined_code = "\n\n".join(snippets)

        # -- optional AI clean -------------------------------------------------
        cleaned_code = self._clean_with_ai(combined_code)
        py_file.write_text(cleaned_code, encoding="utf-8")
        logger.info("Wrote cleaned script: %s", py_file)

        # -- notebook with original cells -------------------------------------
        nb = nbf.v4.new_notebook()
        nb.cells.extend(nbf.v4.new_code_cell(s) for s in snippets)
        nbf.write(nb, str(nb_file))
        logger.info("Wrote notebook:      %s", nb_file)


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