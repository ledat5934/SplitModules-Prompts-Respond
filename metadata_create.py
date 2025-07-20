import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import google.generativeai as genai
from dotenv import load_dotenv

# >>> NEW: dùng DataAnalyzer để quét đệ quy
from data_analyzer import DataAnalyzer, ProjectInfo
# <<< NEW

import time
MAX_RETRY = 3

load_dotenv()  # load GEMINI_API_KEY

def setup_gemini() -> genai.GenerativeModel:
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        sys.exit("  GEMINI_API_KEY / GOOGLE_API_KEY chưa thiết lập trong .env")
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(
        model_name="gemini-2.5-flash",
        generation_config={
            "temperature": 0,
            "max_output_tokens": 4096,
        },
    )

def call_gemini(model: genai.GenerativeModel, prompt: str, retries: int = MAX_RETRY) -> str | None:
    """
    Gọi Gemini và trả về text; thử lại tối đa *retries* lần.
    Trả về None nếu mọi lần đều thất bại hoặc resp.text trống.
    """
    for attempt in range(1, retries + 1):
        try:
            resp = model.generate_content(
                prompt,
                generation_config={"response_mime_type": "application/json"},
            )
            if resp.text and resp.text.strip():
                return resp.text
            print(f"  ⚠ Empty response from Gemini (attempt {attempt}/{retries})")
        except Exception as e:
            print(f"  Gemini request failed (attempt {attempt}/{retries}): {e}")
        if attempt < retries:
            time.sleep(1)
    return None

def safe_json_load(txt: str | None):
    """
    Parse JSON hoặc trả về None nếu không thể.
    """
    if not txt:
        return None
    txt = txt.strip()
    if txt.startswith("```"):
        txt = txt.split("```")[1] if "```" in txt else txt
    try:
        return json.loads(txt)
    except json.JSONDecodeError as e:
        print(f"  JSON parse error, trying quick fix: {e}")
        # quick-n-dirty fixes
        txt = txt.replace("\t", " ").replace("\r", "")
        txt = txt.strip().rstrip(",")  # trailing comma
        try:
            return json.loads(txt)
        except Exception:
            return None

def scan_files(root: Path) -> Dict[str, str]:
    mapping = {}
    for p in root.rglob("*"):
        if p.is_file():
            rel = p.relative_to(root).as_posix()
            mapping[rel] = str(p.resolve())
    return mapping

META_SCHEMA_KEYS = {
    "id", "name", "task", "input_data", "output_data",
    "data file description", "link to the dataset", "files"
}

# ---------- NEW  helper -------------------------------------------------
import re
SUPPORTED_EXTS = (
    ".csv .tsv .txt .json .jsonl .xlsx .xls .parquet .zip"
).split()


def _filter_files_by_description(
    files_map: Dict[str, str],
    description_text: str,
    gemini_json: Dict | None,
) -> Dict[str, str]:
    """
    Giữ lại chỉ những path mà tên file (basename) xuất hiện trong
    description.txt hoặc trong gemini_json['data file description'].
    Nếu không tìm thấy path nào khớp → trả về files_map gốc.
    """
    ref_text = description_text.lower()

    # 1) Bổ sung chuỗi từ khóa 'data file description' của Gemini (nếu có)
    if gemini_json:
        dfd = gemini_json.get("data file description")
        if isinstance(dfd, str):
            ref_text += " " + dfd.lower()
        elif isinstance(dfd, dict):
            ref_text += " " + " ".join(
                f"{k} {v}" for k, v in dfd.items()
            ).lower()

    # 2) Trích xuất tên file trong ref_text theo pattern *.ext
    mentioned: set[str] = set(
        re.findall(
            r"([\w\-.]+(?:"
            + "|".join(re.escape(ext) for ext in SUPPORTED_EXTS)
            + r"))",
            ref_text,
        )
    )

    # 3) Lọc
    filtered = {
        rel: abs_path
        for rel, abs_path in files_map.items()
        if Path(rel).name.lower() in mentioned
    }

    # Nếu Gemini/description không chứa file cụ thể → trả về nguyên danh sách
    return filtered or files_map
# ------------------------------------------------------------------------


def build_entry(
    gemini_json: Dict,
                files_map: Dict[str, str],
    new_id: int,
    description_text: str,
) -> Dict:
    """
    Ghép kết quả Gemini + danh sách file *đã lọc* thành 1 entry chuẩn schema.
    """
    # --- FILTER --------------------------------------------------------
    files_map = _filter_files_by_description(files_map, description_text, gemini_json)

    entry = {
        "id": new_id,
             "link to the dataset": list(files_map.values()),
        "files": files_map,
    }

    # copy các trường do Gemini sinh hợp lệ với schema
    for k, v in gemini_json.items():
        if k in META_SCHEMA_KEYS:
            entry[k] = v

    # Giới hạn 'data file description' theo file đã lọc (nếu Gemini trả dict)
    if isinstance(entry.get("data file description"), dict):
        dfd = entry["data file description"]
        keep_keys = {
            k for k in dfd.keys() if any(name in k for name in mentioned)
        }
        entry["data file description"] = {k: dfd[k] for k in keep_keys}

    entry.setdefault("name", f"dataset_{new_id}")
    return entry
# ----------------- (hết helper) ----------------------------------------

# >>> NEW: tiện ích chuyển ProjectInfo -> files_map
def _files_map_from_project(proj: ProjectInfo) -> Dict[str, str]:
    """
    Convert ProjectInfo.data_files thành mapping {relative_path: absolute_path}
    phù hợp với build_entry().
    """
    mapping: Dict[str, str] = {}
    for lst in proj.data_files.values():
        for p in lst:
            rel = p.relative_to(proj.project_dir).as_posix()
            mapping[rel] = str(p.resolve())
    return mapping
# <<< NEW


def main():
    parser = argparse.ArgumentParser(
        description="Append dataset info to meta-data.json (single folder OR recursive scan)."
    )
    parser.add_argument(
        "root_path",
        help="Folder dataset (giống cách cũ) HOẶC thư mục gốc chứa nhiều project con.",
    )
    parser.add_argument("--meta", default="meta-data.json", help="File meta-data.json")
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Quét root_path đệ quy, tự động thêm tất cả project tìm thấy",
    )
    args = parser.parse_args()

    root_path = Path(args.root_path).resolve()
    if not root_path.exists():
        sys.exit(f"  Path not found: {root_path}")

    # ------------------------------------------------------------------
    #  Đọc (hoặc tạo mới) meta-data.json
    # ------------------------------------------------------------------
    meta_path = Path(args.meta).resolve()
    if meta_path.exists():
        meta_data: List[Dict] = json.loads(meta_path.read_text(encoding="utf-8"))
    else:
        meta_data = []

    # Sử dụng chung biến đếm id cho cả hai chế độ
    existing_ids = [
        int(item["id"]) for item in meta_data if str(item.get("id", "")).isdigit()
    ]
    next_id = (max(existing_ids) if existing_ids else 0) + 1

    # ------------------------------------------------------------------
    #  CHẾ ĐỘ RECURSIVE
    # ------------------------------------------------------------------
    if args.recursive:
        analyzer = DataAnalyzer()
        projects = analyzer.analyze(root_path)
        if not projects:
            sys.exit("  Không tìm thấy description.txt nào trong cây thư mục.")

        model = setup_gemini()  # tạo 1 model dùng chung

        for proj in projects:
            description_text = proj.desc_path.read_text(encoding="utf-8")
            files_map = _files_map_from_project(proj)

            # Chỉ gửi 50 tên path đầu tiên để hạn chế token
            file_list_snippet = "\n".join(list(files_map.keys())[:50])
            prompt = f"""
You are a data-set analyst. From the FREE-TEXT description and the partial
file list below, create a JSON object that follows EXACTLY this schema:

{{
  "name": str,
  "task": str,
  "input_data": str,
  "output_data": str,
  "data file description": str
}}

Return ONLY valid JSON (no markdown).

--- description.txt ---
{description_text}

--- example file list ({len(files_map)} files, first 50) ---
{file_list_snippet}
"""
            gemini_raw = call_gemini(model, prompt)
            gemini_json = safe_json_load(gemini_raw)

            if gemini_json:
                meta_entry = build_entry(gemini_json, files_map, next_id, description_text)
                meta_data.append(meta_entry)
                print(f"   Added project '{proj.name}' (id={next_id})")
            else:
                print("   Gemini failed → creating MINIMAL entry")
                gemini_json = {"name": proj.name, "task": "", "input_data": "",
                               "output_data": "", "data file description": ""}
                meta_entry = build_entry(gemini_json, files_map, next_id, description_text)
                meta_data.append(meta_entry)
                print(f"   Added minimal project entry for '{proj.name}' (id={next_id})")

            next_id += 1  # tăng ID cho project kế tiếp

        # Ghi ra file sau khi xử lý tất cả
        meta_path.write_text(
            json.dumps(meta_data, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"  meta-data.json updated – total {len(meta_data)} entries")
        return

    # ------------------------------------------------------------------
    #  CHẾ ĐỘ CŨ – Xử lý 1 folder dataset
    # ------------------------------------------------------------------
    if not root_path.is_dir():
        sys.exit(f" Folder not found: {root_path}")

    desc_file = root_path / "description.txt"
    if not desc_file.exists():
        sys.exit("  description.txt không tồn tại trong folder dataset.")

    description_text = desc_file.read_text(encoding="utf-8")
    files_map = scan_files(root_path)

    model = setup_gemini()
    file_list_snippet = "\n".join(list(files_map.keys())[:50])
    prompt = f"""
You are a data-set analyst. From the FREE-TEXT description and the partial
file list below, create a JSON object that follows EXACTLY this schema:

{{
  "name": str,
  "task": str,
  "input_data": str,
  "output_data": str,
  "data file description": str           
}}

Return ONLY valid JSON (no markdown).

--- description.txt ---
{description_text}

--- example file list ({len(files_map)} files, first 50) ---
{file_list_snippet}
"""
    gemini_raw = call_gemini(model, prompt)
    gemini_json = safe_json_load(gemini_raw)

    if gemini_json:
        meta_entry = build_entry(gemini_json, files_map, next_id, description_text)
        meta_data.append(meta_entry)
        print(f"  Added dataset id={next_id} to {meta_path}")
    else:
        print(f" Could not parse Gemini response for dataset. Falling back to minimal entry.")
        meta_entry = {"id": next_id, "name": f"dataset_{next_id}", "files": files_map}
    meta_data.append(meta_entry)
    print(f"  Added minimal dataset entry to {meta_path}")


if __name__ == "__main__":
    main()