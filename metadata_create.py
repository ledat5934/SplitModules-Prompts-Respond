import argparse
import json
import os
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List

import google.generativeai as genai
from dotenv import load_dotenv

# optional project helpers ------------------------------------------------
try:
    from data_analyzer import DataAnalyzer  # để dùng ở chế độ --recursive
except Exception:
    DataAnalyzer = None  # type: ignore

load_dotenv()
MAX_RETRY = 3

# ------------------------------------------------------------------ utils
SUPPORTED_EXTS = {
    ".csv", ".tsv", ".txt", ".json", ".jsonl",
    ".xlsx", ".xls", ".parquet",
    ".jpg", ".jpeg", ".png", ".bmp"
}


# ----------------------------------------------------------
# BỎ QUA (SKIP) description.txt khi build lookup
# ----------------------------------------------------------
DESC_HASH = "description.txt"      # ‘hashcode’ nhận diện – thực ra là tên file

def _build_lookup(root: Path) -> Dict[str, Path]:
    lookup: Dict[str, Path] = {}
    for p in root.rglob("*"):
        # 1) bỏ qua description.txt
        if p.is_file() and p.name.lower() == DESC_HASH:
            continue
        # 2) (tùy chọn) bỏ file ẩn bắt đầu bằng dấu chấm
        if p.name.startswith("."):
            continue

        if p.is_file():
            lookup[p.name.lower()] = p.resolve()
        elif p.is_dir():
            try:
                if any(f.is_file() and f.suffix.lower() in SUPPORTED_EXTS
                       for f in p.iterdir()):
                    lookup[p.name.lower()] = p.resolve()
            except PermissionError:
                pass
    return lookup


def _map_paths(llm_json: Dict, lookup: Dict[str, Path]) -> Dict[str, str]:
    """
    Với mỗi key trong 'data file description' (dict) → gán absolute path.
    Chỉ những key tìm thấy mới được giữ lại.
    """
    result: OrderedDict[str, str] = OrderedDict()
    dfd = llm_json.get("data file description", {})
    if not isinstance(dfd, dict):
        return result

    for key in dfd.keys():
        base = key.strip().split("/")[-1].lower()
        if base in lookup:
            result[key] = str(lookup[base])
    return result


# ------------------------------------------------------------------ gemini
def setup_gemini() -> genai.GenerativeModel:
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        sys.exit("GEMINI_API_KEY / GOOGLE_API_KEY chưa thiết lập trong .env")
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(
        model_name="gemini-2.5-flash",
        generation_config={"temperature": 0, "max_output_tokens": 4096},
    )


def call_gemini(model: genai.GenerativeModel, prompt: str, retries: int = MAX_RETRY) -> str | None:
    for i in range(1, retries + 1):
        try:
            resp = model.generate_content(
                prompt, generation_config={"response_mime_type": "application/json"}
            )
            if resp.text and resp.text.strip():
                return resp.text
        except Exception as e:
            print(f"Gemini failed ({i}/{retries}): {e}")
        if i < retries:
            time.sleep(1)
    return None


def safe_json_load(raw: str | None) -> Dict:
    if not raw:
        return {}
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1] if "```" in raw else raw
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # quick repair
        raw = raw.replace("\t", " ").replace("\r", "").strip().rstrip(",")
        try:
            return json.loads(raw)
        except Exception:
            return {}


# ------------------------------------------------------------------ entry
META_SCHEMA_KEYS = {
    "id", "name", "task", "input_data", "output_data",
    "data file description", "link to the dataset", "files"
}


def build_entry(llm_json: Dict, lookup: Dict[str, Path], new_id: int) -> Dict:
    files_map = _map_paths(llm_json, lookup)

    # fallback nếu không match gì → lấy ≤20 path đầu tiên
    if not files_map:
        files_map = {k: str(v) for k, v in list(lookup.items())[:20]}

    entry: Dict = {
        "id": new_id,
        "link to the dataset": list(files_map.values()),
        "files": files_map,
    }
    for k in META_SCHEMA_KEYS:
        if k in llm_json:
            entry[k] = llm_json[k]
    entry.setdefault("name", f"dataset_{new_id}")
    return entry


# ------------------------------------------------------------------ main
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Append dataset info to meta-data.json (single or recursive)."
    )
    parser.add_argument("root", help="Dataset folder OR root containing many projects")
    parser.add_argument("--meta", default="meta-data.json", help="meta-data.json file")
    parser.add_argument("--recursive", action="store_true", help="Recursively scan")
    args = parser.parse_args()

    root_path = Path(args.root).resolve()
    if not root_path.exists():
        sys.exit(f"Path not found: {root_path}")

    meta_path = Path(args.meta).resolve()
    meta_data: List[Dict] = (
        json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else []
    )
    next_id = (
        max(int(it["id"]) for it in meta_data if str(it.get("id", "")).isdigit()) + 1
        if meta_data else 1
    )

    model = setup_gemini()

    # ---------- recursive mode ---------------------------------------
    if args.recursive:
        if not DataAnalyzer:
            sys.exit("DataAnalyzer module not available for --recursive mode.")

        analyzer = DataAnalyzer()
        projects = analyzer.analyze(root_path)
        if not projects:
            sys.exit("No description.txt found under root.")

        for proj in projects:
            print(f"\n Processing project '{proj.name}'")
            lookup = _build_lookup(proj.project_dir)

            file_list_snippet = "\n".join(list(lookup.keys())[:50])
            description_text = proj.desc_path.read_text(encoding="utf-8")
            prompt = f"""
You are a data-set analyst. From the FREE-TEXT description and the partial
file/folder list below, create a JSON object EXACTLY in this schema:

{{
  "name": str,
  "task": str,
  "input_data": str,
  "output_data": str,
  "data file description": {{
     "<file_or_folder_name>": "description"
  }}
}}

Return ONLY valid JSON (no markdown).

--- description.txt ---
{description_text}

--- example list ({len(lookup)} items, first 50) ---
{file_list_snippet}
"""
            llm_raw = call_gemini(model, prompt)
            llm_json = safe_json_load(llm_raw)
            entry = build_entry(llm_json, lookup, next_id)
            meta_data.append(entry)
            print(f"   Added id={next_id}")
            next_id += 1

        # write once
        meta_path.write_text(json.dumps(meta_data, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n meta-data.json updated to {meta_path}")
        return

    # ---------- single-folder mode ----------------------------------
    if not root_path.is_dir():
        sys.exit("Given root is not a directory (single-folder mode).")

    lookup = _build_lookup(root_path)
    file_list_snippet = "\n".join(list(lookup.keys())[:50])
    desc_file = root_path / "description.txt"
    if not desc_file.exists():
        sys.exit("description.txt not found in dataset folder")

    description_text = desc_file.read_text(encoding="utf-8")
    prompt = f"""
You are a data-set analyst. From the FREE-TEXT description and the partial
file/folder list below, create a JSON object EXACTLY in this schema:

{{
  "name": str,
  "task": str,
  "input_data": str,
  "output_data": str,
  "data file description": {{
     "<file_or_folder_name>": "description"
  }}
}}

Return ONLY valid JSON (no markdown).

--- description.txt ---
{description_text}

--- example list ({len(lookup)} items, first 50) ---
{file_list_snippet}
"""
    llm_raw = call_gemini(model, prompt)
    llm_json = safe_json_load(llm_raw)
    entry = build_entry(llm_json, lookup, next_id)
    meta_data.append(entry)

    meta_path.write_text(json.dumps(meta_data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n Added dataset id={next_id} to {meta_path}")


if __name__ == "__main__":
    main()