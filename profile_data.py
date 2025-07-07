import json
from pathlib import Path
from typing import List, Dict
import pandas as pd
from tqdm import tqdm
from ydata_profiling import ProfileReport


def load_metadata(metadata_path: Path) -> List[Dict]:
    try:
        data = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError("Metadata must be a list of dictionaries")
        return data
    except json.JSONDecodeError as e:
        print(f"❌ Lỗi đọc metadata: {e}")
        return []


def check_paths(paths: List[str]) -> Dict:
    exists, missing = [], []
    for p in paths:
        pth = Path(p)
        (exists if pth.exists() else missing).append(p)
    return {"exists": exists, "missing": missing}


import json

def filter_value_counts(profile_json_str: str) -> str:
    try:
        profile_dict = json.loads(profile_json_str)
    except json.JSONDecodeError as e:
        print(f"  ❌ JSON decode error: {e}")
        return profile_json_str

    if "variables" not in profile_dict:
        print("  ⚠️ Không tìm thấy phần 'variables'")
        return profile_json_str

    for var_name, var_info in profile_dict["variables"].items():
        var_type = var_info.get("type", "")
        n_unique = var_info.get("n_unique", 0)

        # Nếu là kiểu văn bản hoặc có quá nhiều giá trị rời rạc → xoá các phần nặng
        should_remove = (
            var_type in ["Text", "Numeric", "Date", "DateTime", "Time", "URL", "Path"]
            or (var_type == "Categorical" and n_unique > 50)
        )

        if should_remove:
            keys_to_remove = [
                "value_counts_without_nan",
                "value_counts_index_sorted",
                "histogram",
                "length_histogram",
                "histogram_length",
                "block_alias_char_counts",
                "word_counts",
                "category_alias_char_counts",
                "script_char_counts",
                "block_alias_values",
                "category_alias_values",
                "character_counts",
                "block_alias_counts",
                "script_counts",
                "category_alias_counts",
                "n_block_alias",
                "n_scripts",
                "n_category",
            ]

            for key in keys_to_remove:
                var_info.pop(key, None)

    return json.dumps(profile_dict, ensure_ascii=False, indent=2)



def csv_profile(csv_path: Path, out_dir: Path) -> Dict:
    df = pd.read_csv(csv_path)
    print(f" Đang phân tích {csv_path.name} ({df.shape[0]} hàng, {df.shape[1]} cột)")

    profile = ProfileReport(
        df,
        title=f"Profile - {csv_path.name}",
        minimal=True,
        samples={
            "random": 5
        },
        correlations={
            "auto": {"calculate": False},
            "pearson": {"calculate": False},
            "spearman": {"calculate": False},
            "kendall": {"calculate": False},
            "phi_k": {"calculate": False}
        },
        missing_diagrams={
            "bar": False,
            "matrix": False,
            "heatmap": False,
            "dendrogram": False
        },
        interactions={"targets": []},
        explorative=False,
        progress_bar=False,
        infer_dtypes=True
    )

    out_dir.mkdir(parents=True, exist_ok=True)

    # Lấy json, lọc value_counts
    profile_json_str = profile.to_json()
    filtered_json_str = filter_value_counts(profile_json_str)

    json_file = out_dir / f"{csv_path.stem}_profile.json"
    json_file.write_text(filtered_json_str, encoding="utf-8")

    summary = {
        "file": str(csv_path),
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_pct": df.isnull().mean().round(4).to_dict(),
        "file_size_mb": round(csv_path.stat().st_size / (1024 * 1024), 2)
    }

    summary_file = out_dir / f"{csv_path.stem}_summary.json"
    summary_file.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f" Đã lưu profile cho {csv_path.name}")
    return summary


def run_profiling(meta_file: str = "meta-data.json", output_root="profiling_results"):
    meta_path = Path(meta_file)
    datasets = load_metadata(meta_path)
    out_root = Path(output_root)
    out_root.mkdir(exist_ok=True)

    print(f" Bắt đầu profiling {len(datasets)} dataset(s)")
    print("=" * 60)

    for ds in tqdm(datasets):
        ds_id: str = str(ds.get('id'))
        ds_name: str = ds.get('name', f'dataset_{ds_id}')
        safe_name = ds_name.replace(' ', '_').replace('/', '_')
        ds_out = out_root / f'{ds_id}_{safe_name}'
        ds_out.mkdir(exist_ok=True)

        print(f"\n  Dataset {ds_id}: {ds_name}")
        print("-" * 40)

        # Lấy danh sách file
        if isinstance(ds.get("files"), dict):
            paths_list = list(ds["files"].values())
        else:
            raw = ds.get("link to the dataset", "")
            paths_list = raw if isinstance(raw, list) else [p.strip() for p in raw.split("\\n") if p.strip()]

        struct_stat = check_paths(paths_list)
        csv_summaries = []

        for p in paths_list:
            if p.lower().endswith(".csv") and Path(p).exists():
                csv_summaries.append(csv_profile(Path(p), ds_out))

        overview = {
            "dataset_id": ds_id,
            "name": ds_name,
            "task": ds.get("task"),
            "structure": struct_stat,
            "csv_profiles": csv_summaries,
        }

        (ds_out / "data_profile.json").write_text(
            json.dumps(overview, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )

    print(f"\n Hoàn tất profiling → {out_root.resolve()}")


if __name__ == "__main__":
    run_profiling("meta-data.json", "profiling_results")
