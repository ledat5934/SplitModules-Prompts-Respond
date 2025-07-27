import argparse
import json
import os
import sys
import time
import traceback
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional

from openai import OpenAI
from dotenv import load_dotenv

# --- Cấu hình ---
load_dotenv()
MAX_RETRY = 3
SUPPORTED_EXTS = {
    ".csv", ".tsv", ".txt", ".json", ".jsonl",
    ".xlsx", ".xls", ".parquet",
    ".jpg", ".jpeg", ".png", ".bmp"
}
DESC_HASH = "description.txt"

# --- Các hàm trợ giúp (Giữ nguyên) ---

def _build_lookup(root: Path) -> Dict[str, Path]:
    """Tạo bảng tra cứu file và thư mục."""
    lookup: Dict[str, Path] = {}
    for p in root.rglob("*"):
        if p.is_file() and p.name.lower() == DESC_HASH:
            continue
        if p.name.startswith("."):
            continue
        if p.is_file():
            lookup[p.name.lower()] = p.resolve()
        elif p.is_dir():
            try:
                if any(f.is_file() and f.suffix.lower() in SUPPORTED_EXTS for f in p.iterdir()):
                    lookup[p.name.lower()] = p.resolve()
            except PermissionError:
                pass
    return lookup

def _map_paths(llm_json: Dict, lookup: Dict[str, Path]) -> Dict[str, str]:
    """Ánh xạ tên file từ AI sang đường dẫn tuyệt đối."""
    result: OrderedDict[str, str] = OrderedDict()
    dfd = llm_json.get("data file description", {})
    if not isinstance(dfd, dict):
        return result
    for key in dfd.keys():
        base = key.strip().split("/")[-1].lower()
        if base in lookup:
            result[key] = str(lookup[base])
    return result

def setup_openai() -> OpenAI:
    """Khởi tạo client OpenAI."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY chưa được thiết lập trong .env")
    return OpenAI(api_key=api_key)

def call_openai(client: OpenAI, model_name: str, prompt: str) -> Optional[str]:
    """Gọi API OpenAI với cơ chế thử lại."""
    for i in range(1, MAX_RETRY + 1):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                #temperature=0.0,
            )
            content = response.choices[0].message.content
            if content and content.strip():
                return content
        except Exception as e:
            print(f"Lỗi gọi OpenAI ({i}/{MAX_RETRY}): {e}")
        if i < MAX_RETRY:
            time.sleep(1)
    return None

def safe_json_load(raw: Optional[str]) -> Dict:
    """Phân tích JSON một cách an toàn."""
    if not raw: return {}
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1] if "```" in raw else raw
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        raw = raw.replace("\t", " ").replace("\r", "").strip().rstrip(",")
        try:
            return json.loads(raw)
        except Exception:
            return {}

def build_entry(llm_json: Dict, lookup: Dict[str, Path], new_id: int) -> Dict:
    """Xây dựng một entry metadata hoàn chỉnh."""
    files_map = _map_paths(llm_json, lookup)
    if not files_map:
        files_map = {k: str(v) for k, v in list(lookup.items())[:20]}
    
    entry: Dict = {
        "id": new_id,
        "link to the dataset": list(files_map.values()),
        "files": files_map,
    }
    for k in ["name", "task", "input_data", "output_data", "data file description"]:
        if k in llm_json:
            entry[k] = llm_json[k]
    entry.setdefault("name", f"dataset_{new_id}")
    return entry

# --- HÀM CHÍNH CÓ THỂ IMPORT ---
def generate_metadata_for_path(root_path: Path, meta_file: str = "meta-data.json") -> Optional[str]:
    """
    Quét một thư mục dataset, tạo metadata bằng OpenAI,
    thêm vào file meta-data và trả về ID của dataset mới.

    Args:
        root_path (Path): Đường dẫn đến thư mục dataset.
        meta_file (str): Đường dẫn đến file meta-data.json.

    Returns:
        Optional[str]: ID của dataset mới (dưới dạng chuỗi) nếu thành công, ngược lại là None.
    """
    print(f"--- Bắt đầu tạo Metadata cho: {root_path.name} ---")
    if not root_path.is_dir():
        print(f"Lỗi: Đường dẫn '{root_path}' không phải là một thư mục.")
        return None

    desc_file = root_path / "description.txt"
    if not desc_file.exists():
        print(f"Lỗi: Không tìm thấy file 'description.txt' trong '{root_path}'")
        return None

    try:
        meta_path = Path(meta_file).resolve()
        meta_data: List[Dict] = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else []
        next_id = max((int(it.get("id", 0)) for it in meta_data), default=0) + 1

        client = setup_openai()
        model_name = "o4-mini"
        lookup = _build_lookup(root_path)
        description_text = desc_file.read_text(encoding="utf-8")
        file_list_snippet = "\n".join(list(lookup.keys())[:20])

        # ** HOÀN LẠI PROMPT GỐC **
        prompt_template = """
You are a data-set analyst. From the FREE-TEXT description and the partial
file/folder list below, create a JSON object EXACTLY in this schema:

{{
  "name": "str",
  "task": "str",
  "input_data": "str(description of the input data)",
  "output_data": "str(description of the output data(probality, label, ...))",
  "data file description": {{
    "<file_or_folder_name>": "description"
  }}
}}

Return ONLY a valid JSON object.

--- description.txt ---
{description_text}

--- example list ({num_items} items, first 20) ---
{file_list_snippet}
"""
        prompt = prompt_template.format(
            description_text=description_text,
            num_items=len(lookup),
            file_list_snippet=file_list_snippet
        )

        llm_raw = call_openai(client, model_name, prompt)
        llm_json = safe_json_load(llm_raw)
        if not llm_json:
            print("Không thể tạo metadata từ OpenAI.")
            return None

        entry = build_entry(llm_json, lookup, next_id)
        meta_data.append(entry)
        meta_path.write_text(json.dumps(meta_data, indent=2, ensure_ascii=False), encoding="utf-8")
        
        print(f"Đã thêm dataset id={next_id} vào {meta_path}")
        return str(next_id)

    except Exception as e:
        print(f"Lỗi nghiêm trọng khi tạo metadata: {e}")
        traceback.print_exc()
        return None

def main_cli():
    """Hàm main để chạy script từ dòng lệnh."""
    # NOTE: Chế độ --recursive hiện không được sử dụng trong pipeline chính
    # nhưng được giữ lại để có thể chạy độc lập.
    try:
        from data_analyzer import DataAnalyzer
    except ImportError:
        DataAnalyzer = None

    parser = argparse.ArgumentParser(description="Tạo và thêm metadata vào meta-data.json.")
    parser.add_argument("root", help="Thư mục dataset HOẶC thư mục gốc chứa nhiều project")
    parser.add_argument("--meta", default="meta-data.json", help="File meta-data.json")
    parser.add_argument("--recursive", action="store_true", help="Quét đệ quy để tìm các project")
    args = parser.parse_args()

    root_path = Path(args.root).resolve()
    if not root_path.exists():
        sys.exit(f"Lỗi: Không tìm thấy đường dẫn: {root_path}")

    if not args.recursive:
        # Chế độ một thư mục
        new_id = generate_metadata_for_path(root_path, args.meta)
        if new_id:
            print(f"\nTạo metadata thành công. ID mới: {new_id}")
            sys.exit(0)
        else:
            print("\nTạo metadata thất bại.")
            sys.exit(1)
    else:
        # Chế độ đệ quy (giữ lại từ code gốc)
        if not DataAnalyzer:
            sys.exit("Không tìm thấy module DataAnalyzer cho chế độ --recursive.")
        
        # ... (logic cho chế độ đệ quy có thể được thêm vào đây nếu cần) ...
        print("Chế độ đệ quy chưa được triển khai đầy đủ trong phiên bản này.")
        sys.exit(1)


if __name__ == "__main__":
    main_cli()