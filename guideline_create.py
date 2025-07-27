import json
from pathlib import Path
from typing import Dict, List, Optional
from openai import OpenAI # THAY ĐỔI: Import OpenAI
import os
from datetime import datetime
from dotenv import load_dotenv
import re
import tiktoken # THAY ĐỔI: Import tiktoken để đếm token

# LOAD .ENV FILE
load_dotenv()

# THAY ĐỔI: Hàm đếm token cho các mô hình OpenAI
def count_tokens_openai(text: str, model: str = "o4-mini") -> int:
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("o200k_base")
    return len(encoding.encode(text))

def extract_guideline_input(dataset_profile_dir: Path, meta_data: Dict) -> Dict:
    """
    Extract thông tin từ profiling results (Không thay đổi)
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
    
    summaries = {}
    for summary_file in dataset_profile_dir.glob('*_summary.json'):
        file_name = summary_file.stem.replace('_summary', '')
        summaries[file_name] = json.loads(summary_file.read_text(encoding='utf-8'))
    
    profiles = {}
    for profile_file in dataset_profile_dir.glob('*_profile.json'):
        file_name = profile_file.stem.replace('_profile', '')
        profile_data = json.loads(profile_file.read_text(encoding='utf-8'))
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
    Tạo summary thông minh về variables cho LLM (Không thay đổi)
    """
    if not variables:
        return {}
    MAX_TEXT_LENGTH = 50
    var_types = {"numerical": [], "categorical": [], "text": [], "datetime": [], "other": []}
    
    for var_name, var_info in variables.items():
        var_type = var_info.get("type", "")
        
        first_row_value = var_info.get("first_rows", {}).get("0", "N/A")
        display_row = first_row_value 
        if var_type == "Text":
            first_row_str = str(first_row_value)
            if len(first_row_str) > MAX_TEXT_LENGTH:
                display_row = first_row_str[:MAX_TEXT_LENGTH] + "..."
        
        var_summary = {
            "name": var_name, "type": var_type,
            "missing_pct": round(var_info.get("p_missing", 0), 3),
            "n_distinct": var_info.get("n_distinct", 0),
            "first_row": display_row 
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
            
    def _importance(v):
        return (-v["missing_pct"], -v.get("n_distinct", 0))

    all_vars = [v for vars_list in var_types.values() for v in vars_list]
    top_vars = sorted(all_vars, key=_importance)[:10]

    trimmed_types = {k: [] for k in var_types}
    for v in top_vars:
        t = v["type"]
        if t == "Numeric": trimmed_types["numerical"].append(v)
        elif t == "Categorical": trimmed_types["categorical"].append(v)
        elif t == "Text": trimmed_types["text"].append(v)
        elif t in ("DateTime", "Date", "Time"): trimmed_types["datetime"].append(v)
        else: trimmed_types["other"].append(v)

    return {"summary_stats": summary_stats, "variables_by_type": trimmed_types}

# SỬA LỖI: Hàm này giờ sẽ return đúng prompt mới
def create_enhanced_guideline_prompt(guideline_input: Dict) -> str:
    """
    Tạo prompt nâng cao với các nguyên tắc và ví dụ cụ thể để nhận được guideline chất lượng.
    """
    task_info = guideline_input['task_info']
    summaries = guideline_input['summaries']
    profiles = guideline_input['profiles']
    
    dataset_name = task_info.get('name', 'N/A')
    task_desc = task_info.get('task_description', 'N/A')
    data_file_description = task_info.get('data_file_description', 'N/A')
    
    sample_summary = {}
    if summaries:
        all_train_summaries = [summary for filename, summary in summaries.items() if 'train' in filename.lower()]
        sample_summary = all_train_summaries[0] if all_train_summaries else (list(summaries.values())[0] if summaries else {})
    
    n_rows = sample_summary.get('n_rows', 0)
    n_cols = sample_summary.get('n_cols', 0)
    
    sample_profile = {}
    if profiles:
        all_train_profiles = [profile for filename, profile in profiles.items() if 'train' in filename.lower()]
        sample_profile = all_train_profiles[0] if all_train_profiles else (list(profiles.values())[0] if profiles else {})

    variables = sample_profile.get('variables', {})
    variables_summary_str = json.dumps(create_variables_summary(variables), indent=2, ensure_ascii=False)

    prompt = f"""You are an expert Machine Learning architect. Your task is to analyze the provided dataset information and create a specific, actionable, and justified guideline for an AutoML pipeline.
## Dataset Information:
- **Dataset**: {dataset_name}
- **Task**: {task_desc}
- **Size**: {n_rows:,} rows, {n_cols} columns
- **Data File Description**: {data_file_description}

## Variables Analysis Summary:
```json
{variables_summary_str}
Guideline Generation Principles & Examples
Your response must be guided by the following principles. Refer to these examples to understand the expected level of detail.
BE SPECIFIC AND ACTIONABLE: Your recommendations must be concrete actions.
Bad (Generic): "Handle missing values"
Good (Specific): "Impute 'Age' with the median"
JUSTIFY YOUR CHOICES INTERNALLY: Even though the final JSON doesn't have a reason for every single step, your internal reasoning process must be sound. Base your choices on the data's properties (type, statistics, alerts).
IT'S OKAY TO OMIT: If a step is not necessary, provide an empty list [] or null for that key in the JSON output.
CONSIDER FEATURE SCALING FOR LARGE NUMERIC VALUES: If any numerical feature has a very large mean or standard deviation (e.g., >10,000), consider applying scaling such as StandardScaler or MinMaxScaler. Scale the numerical target if it has a very large mean and then rescale when predicts.
IMPORTANT: When doing Deep Learning task, using lazy loading to reduce memory usage.
High-Quality Examples
Example 1: Feature Engineering for a DateTime column
If you see a DateTime column like 'transaction_date', a good feature_engineering list would be ["Extract 'month' from 'transaction_date'", "Extract 'day_of_week' from 'transaction_date'"].

Example 2: Handling High Cardinality Categorical Data
If a categorical column 'product_id' has over 100 unique values, a good feature_engineering recommendation would be ["Apply frequency encoding to 'product_id'"].

Example 3: Handling Missing Numerical Data
If you see a numeric column 'income' with 25% missing values and a skewed distribution, a good missing_values recommendation would be ["Impute 'income' with its median"].

Required Thinking Process (Do not output this part)
Before generating the final JSON, think step-by-step:

First, carefully identify the target variable and the task type (classification/regression).

Second, review each variable. What are its type, statistics, and potential issues?

Third, based on the data properties, decide on the most appropriate ML or DL algorithm.

Forth, think about the suitable preprocessing for the algorithm.

Consider using a pretrained model for NLP or CV tasks if necessary.

Finally, compile these specific actions into the required JSON format below.

Output Format: Your response must be the JSON format below:
Please provide your response in JSON format. It is acceptable to provide an empty list or null for recommendations if none are suitable.

IMPORTANT: Ensure the generated JSON is perfectly valid.

All strings must be enclosed in double quotes.

No trailing commas.

No comments (// or #) within the JSON output.

{{
"target_identification": {{
"target_variable": "identified_target_column_name",
"reasoning": "explanation for target selection",
"task_type": "classification/regression/etc"
}},
"modeling": {{
"recommended_algorithms": ["algorithm"],
"explanation": "explanation for the recommended algorithms",
"model_selection": ["model_name1", "model_name2"],
"model_selection_reasoning": "explanation for the model selection",
"output_file_structure": {{"submission.csv": "submission file description"}}
}},
"preprocessing": {{
"data_cleaning": ["specific step 1"],
"feature_engineering": ["specific technique 1"],
"explanation": "explanation for the feature engineering",
"missing_values": ["strategy 1"],
"feature_selection": ["method 1"],
"data_splitting": {{"train": 0.8, "val": 0.2}}
}},
"evaluation": {{
"metrics": ["metric 1", "metric 2"],
"validation_strategy": ["approach 1"],
"performance_benchmarking": ["baseline 1"],
"result_interpretation": ["interpretation 1"]
}}
}}"""
    return prompt
def call_openai_for_guideline(prompt: str, model: str = "o4-mini") -> tuple[Optional[str], int, int]:

    input_tokens = count_tokens_openai(prompt, model)
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("  OpenAI API key not found. Please check environment variables or .env file.")
            return None, input_tokens, 0
        client = OpenAI(api_key=api_key)
    
        response = client.chat.completions.create(
            model=model,
            messages=[
            {"role": "user", "content": prompt}
            ],
            #temperature=0, # Giảm nhiệt độ để có kết quả nhất quán
            top_p=0.95,
            max_tokens=4096,  
            response_format={"type": "json_object"} # Yêu cầu OpenAI trả về JSON
        )
    
        response_text = response.choices[0].message.content
        prompt_tokens = response.usage.prompt_tokens
        completion_tokens = response.usage.completion_tokens
        return response_text, prompt_tokens, completion_tokens
    except Exception as e:
        print(f"  Error calling OpenAI: {e}")
        return None, input_tokens, 0
    
def generate_guidelines_for_dataset(guideline_input: Dict, output_dir: Path) -> Optional[Dict]:
    dataset_name = guideline_input['task_info']['name']
    dataset_id = guideline_input['task_info']['dataset_id']
    model_used = "o4-mini"
    print(f" Generating guidelines for: {dataset_name}")
    
    prompt = create_enhanced_guideline_prompt(guideline_input)
    safe_name = dataset_name.replace(" ", "_").replace("/", "_")
    
    prompt_dir = output_dir / "prompts"
    prompt_dir.mkdir(exist_ok=True)
    
    prompt_file = prompt_dir / f"{dataset_id}_{safe_name}_prompt.txt"
    prompt_file.write_text(prompt, encoding="utf-8")
    print(f"  Prompt saved to: {prompt_file}")

    # Gọi OpenAI
    openai_response, input_tokens, output_tokens = call_openai_for_guideline(prompt, model=model_used)

    print(f"  Token Usage: Input: {input_tokens:,}, Output: {output_tokens:,}, Total: {input_tokens + output_tokens:,}")

    # Tính chi phí
    input_cost = (input_tokens / 1_000_000) * 0.15
    output_cost = (output_tokens / 1_000_000) * 0.60
    total_cost = input_cost + output_cost
    print(f"  Estimated Cost: ${total_cost:.6f}")

    # Nếu có phản hồi
    if openai_response:
        guidelines = None
        try:
            # Phân tích JSON
            guidelines = json.loads(openai_response)
            print(f"  Guidelines parsed successfully!")
        except json.JSONDecodeError as e:
            print(f"  JSON parse error: {e}")
            print("  Attempting to fix and re-parse...")
            response_fixed = openai_response.strip().replace("```json", "").replace("```", "")
            response_fixed = re.sub(r",\s*([}\]])", r"\1", response_fixed)
            try:
                guidelines = json.loads(response_fixed)
                print("  Successfully parsed after manual fixing.")
            except json.JSONDecodeError as e2:
                print(f"  Failed to parse even after fixing: {e2}")
                guidelines = {
                    "raw_response": openai_response,
                    "parse_error": str(e2)
                }

        if "target_identification" in guidelines:
            target_info = guidelines["target_identification"]
            print(f"  Target identified: {target_info.get('target_variable', 'N/A')}")

        result = {
            "dataset_info": {
                "id": dataset_id,
                "name": dataset_name,
                "generated_at": datetime.now().isoformat(),
                "model_used": model_used,
                "token_usage": {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens,
                    "estimated_cost": total_cost
                }
            },
            "guidelines": guidelines
        }

        output_file = output_dir / f"{dataset_id}_{safe_name}_guideline.json"
        output_file.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
        print(f"  Guidelines saved to: {output_file.name}")
        return result

    else:
        print(f"  Failed to generate guidelines for {dataset_name}")
        return None
    
def generate_all_guidelines(guideline_inputs: List[Dict], output_dir: str = "guidelines_output") -> List[Dict]:
    """
    Sinh guidelines cho tất cả datasets (Không thay đổi)
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
    Chuẩn bị input cho guideline generation (Không thay đổi)
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
    Main function: Chạy toàn bộ pipeline với OpenAI
    """
    print("AutoML Guideline Generation Pipeline (OpenAI gpt-o4-mini)")  # Cập nhật tên
    print("=" * 50)

    print("\nStep 1: Preparing guideline inputs...")
    guideline_inputs = prepare_for_guideline_generation()

    if not guideline_inputs:
        print("No guideline inputs prepared. Exiting...")
        return

    print("\nStep 2: Generating guidelines using OpenAI...")  # Cập nhật tên
    generate_all_guidelines(guideline_inputs)

    print(f"\nPipeline completed successfully!")

if __name__ == "__main__":
    main()







