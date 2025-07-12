import json
from pathlib import Path
from typing import Dict, List, Optional
import google.generativeai as genai
import os
from datetime import datetime
from dotenv import load_dotenv
import re # Import regex để sửa lỗi JSON

# LOAD .ENV FILE
load_dotenv()

def count_tokens_gemini(text: str) -> int:
    """
    Đếm tokens cho Gemini (approximation)
    """
    # Gemini token counting is different, using approximation
    return len(text.split()) * 1.2

def extract_guideline_input(dataset_profile_dir: Path, meta_data: Dict) -> Dict:
    """
    Extract thông tin từ profiling results
    """
    task_info = {
        'dataset_id': meta_data.get('id'),
        'name': meta_data.get('name'),
        'task_description': meta_data.get('task'),
        'input_description': meta_data.get('input_data'),
        'output_description': meta_data.get('output_data'),
        'data_file_description': meta_data.get('data file description'),
        'link_to_the_dataset': meta_data.get('link to the dataset'),
    }
    
    # Load summaries
    summaries = {}
    for summary_file in dataset_profile_dir.glob('*_summary.json'):
        file_name = summary_file.stem.replace('_summary', '')
        summaries[file_name] = json.loads(summary_file.read_text(encoding='utf-8'))
    
    # Load profiles
    profiles = {}
    for profile_file in dataset_profile_dir.glob('*_profile.json'):
        file_name = profile_file.stem.replace('_profile', '')
        profile_data = json.loads(profile_file.read_text(encoding='utf-8')) # Thêm encoding='utf-8'
        profiles[file_name] = {
            'table': profile_data.get('table', {}),
            'variables': profile_data.get('variables', {}),
            'alerts': profile_data.get('alerts', []),
        }
    
    return {
        'task_info': task_info,
        'summaries': summaries,
        'profiles': profiles,
    }

def create_variables_summary(variables: Dict) -> Dict:
    """
    Tạo summary thông minh về variables cho LLM (không detect target)
    """
    if not variables:
        return {}
    
    var_types = {"numerical": [], "categorical": [], "text": [], "datetime": [], "other": []}
    
    for var_name, var_info in variables.items():
        var_type = var_info.get("type", "")
        var_summary = {
            "name": var_name, "type": var_type,
            "missing_pct": round(var_info.get("p_missing", 0), 3),
            "n_distinct": var_info.get("n_distinct", 0)
        }
        
        if var_type == "Categorical":
            var_summary.update({
                "imbalance": round(var_info.get("imbalance", 0), 3),
                "is_binary": var_info.get("n_distinct", 0) == 2
            })
            if var_info.get("n_distinct", 0) <= 10:
                value_counts = var_info.get("value_counts_without_nan", {})
                if value_counts:
                    var_summary["top_values"] = dict(list(value_counts.items())[:5])
        elif var_type == "Numeric":
            var_summary.update({
                "min": var_info.get("min"), "max": var_info.get("max"),
                "mean": round(var_info.get("mean", 0), 3) if var_info.get("mean") else None,
                "std": round(var_info.get("std", 0), 3) if var_info.get("std") else None
            })
        
        if var_type == "Numeric": var_types["numerical"].append(var_summary)
        elif var_type == "Categorical": var_types["categorical"].append(var_summary)
        elif var_type == "Text": var_types["text"].append(var_summary)
        elif var_type in ["DateTime", "Date", "Time"]: var_types["datetime"].append(var_summary)
        else: var_types["other"].append(var_summary)

    for v_type in var_types:
        var_types[v_type] = sorted(var_types[v_type], key=lambda x: (x["missing_pct"], -x.get("n_distinct", 0)))
        
    summary_stats = {
        "total_variables": len(variables),
        "by_type": {k: len(v) for k, v in var_types.items() if v},
        "missing_data": {
            "variables_with_missing": sum(1 for v in variables.values() if v.get("p_missing", 0) > 0),
            "avg_missing_pct": round(sum(v.get("p_missing", 0) for v in variables.values()) / (len(variables) or 1), 3)
        },
        "data_issues": {"id_like_features": [], "high_cardinality_features": [], "highly_imbalanced_features": []}
    }
    
    total_rows = max((v.get("n", 1) for v in variables.values()), default=1)
    for var_name, var_info in variables.items():
        n_distinct = var_info.get("n_distinct", 0)
        if n_distinct > total_rows * 0.95 and total_rows > 1: summary_stats["data_issues"]["id_like_features"].append(var_name)
        if n_distinct > 100: summary_stats["data_issues"]["high_cardinality_features"].append(var_name)
        if var_info.get("type") == "Categorical" and var_info.get("imbalance", 0) > 0.95:
            summary_stats["data_issues"]["highly_imbalanced_features"].append({"name": var_name, "imbalance": round(var_info.get("imbalance", 0), 3)})
            
    return {"summary_stats": summary_stats, "variables_by_type": var_types}


# --- START: PROMPT ĐÃ ĐƯỢC CẬP NHẬT ---
def create_enhanced_guideline_prompt(guideline_input: Dict) -> str:
    """
    Tạo prompt nâng cao với các nguyên tắc và ví dụ cụ thể để nhận được guideline chất lượng.
    """
    task_info = guideline_input['task_info']
    summaries = guideline_input['summaries']
    profiles = guideline_input['profiles']
    
    dataset_name = task_info.get('name', 'N/A')
    input_data = task_info.get('input_description', 'N/A')
    output_data = task_info.get('output_description', 'N/A')
    data_file_description = task_info.get('data_file_description', 'N/A')
    task_desc = task_info.get('task_description', 'N/A')
    
    sample_summary = {}
    if summaries:
        # Tìm summary đầu tiên có chữ 'train' trong tên file (key), không phân biệt hoa thường
        # Nếu không tìm thấy, sẽ tự động lấy summary đầu tiên trong danh sách làm dự phòng
        sample_summary = next(
            (summary for filename, summary in summaries.items() if 'train' in filename.lower()),
            list(summaries.values())[0]
        )
    
    # Lấy n_rows và n_cols từ summary đã chọn
    n_rows = sample_summary.get('n_rows', 0)
    n_cols = sample_summary.get('n_cols', 0)

    
    sample_profile = list(profiles.values())[0] if profiles else {}
    alerts = sample_profile.get('alerts', [])
    variables = sample_profile.get('variables', {})
    
    variables_summary_str = json.dumps(create_variables_summary(variables), indent=2, ensure_ascii=False)

    prompt = f"""You are an expert Machine Learning architect. Your task is to analyze the provided dataset information and create a specific, actionable, and justified guideline for an AutoML pipeline.
## Dataset Information:
- **Dataset**: {dataset_name}
- **Task**: {task_desc}
- **Size**: {n_rows:,} rows, {n_cols} columns
- **Key Quality Alerts**: {alerts[:3] if alerts else 'None'}
- **Data File Description**: {data_file_description}


## Variables Analysis Summary:
```json
{variables_summary_str}
```

## Guideline Generation Principles & Examples
Your response must be guided by the following principles. Refer to these examples to understand the expected level of detail.

**BE SPECIFIC AND ACTIONABLE**: Your recommendations must be concrete actions.
-  Bad (Generic): "Handle missing values"
-  Good (Specific): "Impute 'Age' with the median"

**JUSTIFY YOUR CHOICES INTERNALLY**: Even though the final JSON doesn't have a reason for every single step, your internal reasoning process must be sound. Base your choices on the data's properties (type, statistics, alerts).

**IT'S OKAY TO OMIT**: If a step is not necessary (e.g., feature selection for a dataset with very few features), provide an empty list [] or null for that key in the JSON output.
**CONSIDER FEATURE SCALING FOR LARGE NUMERIC VALUES**:  
If any numerical feature (including the target variable) has a very large mean or standard deviation (e.g., >10,000), consider applying scaling such as StandardScaler or MinMaxScaler.
## High-Quality Examples

**Example 1: Feature Engineering for a DateTime column**
If you see a DateTime column like 'transaction_date', a good feature_engineering list would be ["Extract 'month' from 'transaction_date'", "Extract 'day_of_week' from 'transaction_date'"].

**Example 2: Handling High Cardinality Categorical Data**
If a categorical column 'product_id' has over 100 unique values, a good feature_engineering recommendation would be ["Apply frequency encoding to 'product_id'"] instead of one-hot encoding to avoid a memory explosion.

**Example 3: Handling Missing Numerical Data**
If you see a numeric column 'income' with 25% missing values and a skewed distribution, a good missing_values recommendation would be ["Impute 'income' with its median"].

## Required Thinking Process (Do not output this part)
Before generating the final JSON, think step-by-step:
1. First, carefully identify the target variable and the task type (classification/regression).
2. Second, review each variable. What are its type, statistics, and potential issues?
3. Third, based on the data properties and the examples above, decide on the most appropriate, specific ML or DL algorithm for this task.
4. Forth, think the suitable preprocessing for the algorithm(Example: If use pretrained model for NLP tasks, feature engineering should not have 'generate embedding' step).
4. Consider using pretrained model for NLP or CV tasks if necessary.
5. If use pretrained model, choose most appropriate models for the task.
6. With text data, consider between pretrained model or BOW, TF-IDF, ... base on task.
7. Finally, compile these specific actions into the required JSON format below.

## Output Format: Your response must be the JSON format below:
Please provide your response in JSON format. It is acceptable to provide an empty list or null for recommendations if none are suitable.

**IMPORTANT**: Ensure the generated JSON is perfectly valid.
- All strings must be enclosed in double quotes.
- All backslashes inside strings must be properly escaped (e.g., "C:\\\\path" not "C:\\path").
- There should be no unescaped newline characters within a string value.
- Do not add trailing commas.
- Do not include comments (// or #) within the JSON output.

{{
    "target_identification": {{
        "target_variable": "identified_target_column_name",
        "reasoning": "explanation for target selection",
        "task_type": "classification/regression/etc"
    }},
    "modeling": {{
        "recommended_algorithms": ["algorithm"],
        "explanation": "explanation for the recommended algorithms",
        "model_selection": [model_name1, model_name2](description: name of the pretrained model if using, if not using, leave it blank),
        "model_selection_reasoning": "explanation for the model selection",
        "output_file_structure": {{"submission.csv": "submission file for the test dataset, contain n Columns:[...], have the same columns but not the same rows with sample_submission.csv"}}
    }},
    "preprocessing": {{
        "data_cleaning": ["specific step 1", "specific step 2"],
        "feature_engineering": ["specific technique 1", "specific technique 2"],
        "explanation": "explanation for the feature engineering",
        "missing_values": ["strategy 1", "strategy 2"],
        "feature_selection": ["method 1", "method 2"],
        "data_splitting": {{"train": 0.8, "val": 0.2, "strategy": "stratified"}}
    }},
    "evaluation": {{
        "metrics": ["metric 1", "metric 2"],
        "validation_strategy": ["approach 1", "approach 2"],
        "performance_benchmarking": ["baseline 1", "baseline 2"],
        "result_interpretation": ["interpretation 1", "interpretation 2"]
    }}
}}"""

    #  FIX: Chỉ return prompt trực tiếp, không dùng .format() nữa
    # Vì f-string đã thay thế tất cả variables rồi
    return prompt

# --- END: PROMPT ĐÃ ĐƯỢC CẬP NHẬT ---


def call_gemini_for_guideline(prompt: str, model: str = "gemini-2.5-flash") -> tuple[Optional[str], int, int]:
    """
    Gọi Gemini để sinh guideline
    Returns: (response, input_tokens, output_tokens)
    """
    input_tokens = count_tokens_gemini(prompt)
    try:
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            print("  Gemini API key not found. Please check environment variables or .env file.")
            return None, input_tokens, 0
            
        genai.configure(api_key=api_key)
        model_instance = genai.GenerativeModel(
            model_name=model,
            generation_config={
                "temperature": 0, # Giảm nhiệt độ để có kết quả nhất quán hơn
                "top_p": 0.95, "top_k": 40,
                "max_output_tokens": 8000,
                "response_mime_type": "application/json", # Yêu cầu Gemini trả về JSON
            }
        )
        response = model_instance.generate_content(prompt)
        
        if response.text:
            output_tokens = count_tokens_gemini(response.text)
            return response.text, input_tokens, output_tokens
        else:
            print(f"  No response text from Gemini.")
            return None, input_tokens, 0
    
    except Exception as e:
        print(f"  Error calling Gemini: {e}")
        return None, input_tokens, 0

def generate_guidelines_for_dataset(guideline_input: Dict, output_dir: Path) -> Optional[Dict]:
    """
    Sinh guideline cho 1 dataset
    """
    dataset_name = guideline_input['task_info']['name']
    dataset_id = guideline_input['task_info']['dataset_id']
    
    print(f" Generating guidelines for: {dataset_name}")
    
    prompt = create_enhanced_guideline_prompt(guideline_input)
    gemini_response, input_tokens, output_tokens = call_gemini_for_guideline(prompt)
    
    print(f"  Token Usage: Input: {input_tokens:,}, Output: {output_tokens:,}, Total: {input_tokens + output_tokens:,}")
    total_cost = (input_tokens * 0.00015 / 1000) + (output_tokens * 0.0006 / 1000)
    print(f"  Estimated Cost: ${total_cost:.6f}")
    
    if gemini_response:
        guidelines = None
        # --- START: CẢI THIỆN LOGIC PARSE JSON ---
        try:
            # Vì đã yêu cầu response_mime_type='application/json', Gemini thường trả về JSON hợp lệ.
            # Không cần phải xóa ```json nữa.
            guidelines = json.loads(gemini_response)
            print(f"  Guidelines parsed successfully!")
            
        except json.JSONDecodeError as e:
            print(f"  JSON parse error: {e}")
            print("  Attempting to fix and re-parse...")
            # Fallback: cố gắng sửa lỗi LLM tự thêm vào
            response_fixed = gemini_response.strip()
            if response_fixed.startswith("```json"):
                response_fixed = response_fixed[7:]
            if response_fixed.endswith("```"):
                response_fixed = response_fixed[:-3]
            response_fixed = re.sub(r",\s*([}\]])", r"\1", response_fixed) # Xóa trailing comma

            try:
                guidelines = json.loads(response_fixed)
                print("  Successfully parsed after manual fixing.")
            except json.JSONDecodeError as e2:
                print(f"  Failed to parse even after fixing: {e2}")
                guidelines = {
                    "raw_response": gemini_response,
                    "parse_error": str(e2)
                }
        # --- END: CẢI THIỆN LOGIC PARSE JSON ---

        if "target_identification" in guidelines:
            target_info = guidelines["target_identification"]
            print(f"  Target identified: {target_info.get('target_variable', 'N/A')}")
        
        result = {
            "dataset_info": {
                "id": dataset_id, "name": dataset_name,
                "generated_at": datetime.now().isoformat(), "model_used": "gemini-2.5-flash",
                "token_usage": {
                    "input_tokens": input_tokens, "output_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens, "estimated_cost": total_cost
                }
            },
            "guidelines": guidelines
        }
        
        safe_name = dataset_name.replace(' ', '_').replace('/', '_')
        output_file = output_dir / f"{dataset_id}_{safe_name}_guideline.json"
        output_file.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
        print(f"  Guidelines saved to: {output_file.name}")
        return result
    
    else:
        print(f"  Failed to generate guidelines for {dataset_name}")
        return None

def generate_all_guidelines(guideline_inputs: List[Dict], output_dir: str = "guidelines_output") -> List[Dict]:
    """
    Sinh guidelines cho tất cả datasets
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    all_guidelines, total_input_tokens, total_output_tokens, total_cost = [], 0, 0, 0
    
    print(f"Generating guidelines for {len(guideline_inputs)} datasets...")
    print("=" * 70)
    
    for i, guideline_input in enumerate(guideline_inputs, 1):
        print(f"\nDataset {i}/{len(guideline_inputs)}")
        print("-" * 40)
        guideline_result = generate_guidelines_for_dataset(guideline_input, output_path)
        if guideline_result:
            all_guidelines.append(guideline_result)
            token_usage = guideline_result["dataset_info"]["token_usage"]
            total_input_tokens += token_usage["input_tokens"]
            total_output_tokens += token_usage["output_tokens"]
            total_cost += token_usage["estimated_cost"]
            
    consolidated_file = output_path / "all_guidelines.json"
    consolidated_file.write_text(json.dumps(all_guidelines, indent=2, ensure_ascii=False), encoding='utf-8')
    
    print("\n" + "=" * 70)
    print(f"All guidelines generated and saved to '{output_dir}' directory.")
    print(f"Consolidated file: {consolidated_file}")
    print(f"Total datasets processed: {len(all_guidelines)}")
    print("\nTOTAL TOKEN USAGE:")
    print(f"  Input: {total_input_tokens:,}, Output: {total_output_tokens:,}, Total: {total_input_tokens + total_output_tokens:,}")
    print(f"  Total estimated cost: ${total_cost:.6f}")
    
    return all_guidelines

def prepare_for_guideline_generation(meta_file: str = 'meta-data.json', profiling_root: str = 'profiling_results', output_file: str = 'guideline_input.json'):
    """
    Chuẩn bị input cho guideline generation
    """
    meta_data = json.loads(Path(meta_file).read_text(encoding='utf-8'))
    profiling_dir = Path(profiling_root)
    guideline_inputs = []
    
    print(f"Extracting inputs for {len(meta_data)} datasets...")
    for dataset in meta_data:
        ds_id = str(dataset.get('id'))
        ds_name = dataset.get('name', f'dataset_{ds_id}')
        safe_name = ds_name.replace(' ', '_').replace('/', '_')
        dataset_dir = profiling_dir / f'{ds_id}_{safe_name}'
        
        if dataset_dir.exists():
            print(f" Processing {ds_name}...")
            guideline_input = extract_guideline_input(dataset_dir, dataset)
            guideline_inputs.append(guideline_input)
        else:
            print(f"  Skip {ds_name} - profiling directory not found at {dataset_dir}")
            
    output_path = Path(output_file)
    output_path.write_text(json.dumps(guideline_inputs, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f"\nSaved {len(guideline_inputs)} guideline inputs to: {output_path}")
    return guideline_inputs

def main():
    """
    Main function: Chạy toàn bộ pipeline
    """
    print("AutoML Guideline Generation Pipeline (Gemini)")
    print("=" * 50)
    
    print("\nStep 1: Preparing guideline inputs...")
    guideline_inputs = prepare_for_guideline_generation()
    
    if not guideline_inputs:
        print("No guideline inputs prepared. Exiting...")
        return
        
    print("\nStep 2: Generating guidelines using Gemini...")
    generate_all_guidelines(guideline_inputs)
    
    print(f"\nPipeline completed successfully!")

if __name__ == "__main__":
    main()
