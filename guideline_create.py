import json
from pathlib import Path
from typing import Dict, List, Optional
import google.generativeai as genai  # pip install google-generativeai
import os
from datetime import datetime
from dotenv import load_dotenv  # pip install python-dotenv

# 🔧 LOAD .ENV FILE
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
        profile_data = json.loads(profile_file.read_text())
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

def create_variables_summary(variables: Dict, max_detail_vars: int = 10) -> Dict:
    """
    Tạo summary thông minh về variables cho LLM (không detect target)
    """
    
    if not variables:
        return {}
    
    # Phân loại variables
    var_types = {
        "numerical": [],
        "categorical": [],
        "text": [],
        "datetime": [],
        "other": []
    }
    
    # Phân tích từng variable
    for var_name, var_info in variables.items():
        var_type = var_info.get("type", "")
        
        var_summary = {
            "name": var_name,
            "type": var_type,
            "missing_pct": round(var_info.get("p_missing", 0), 3),
            "n_distinct": var_info.get("n_distinct", 0),
            "n_unique": var_info.get("n_unique", 0)
        }
        
        # Thêm info specific cho từng type
        if var_type == "Categorical":
            var_summary.update({
                "imbalance": round(var_info.get("imbalance", 0), 3),
                "is_binary": var_info.get("n_distinct", 0) == 2
            })
            
            # Thêm value_counts cho categorical quan trọng
            if var_info.get("n_distinct", 0) <= 10:  # Low cardinality
                value_counts = var_info.get("value_counts_without_nan", {})
                if value_counts:
                    # Chỉ lấy top 5 values
                    top_values = dict(list(value_counts.items())[:5])
                    var_summary["top_values"] = top_values
                
        elif var_type == "Numeric":
            var_summary.update({
                "min": var_info.get("min"),
                "max": var_info.get("max"),
                "mean": round(var_info.get("mean", 0), 3) if var_info.get("mean") else None,
                "std": round(var_info.get("std", 0), 3) if var_info.get("std") else None
            })
        
        # Phân loại vào type
        if var_type == "Numeric":
            var_types["numerical"].append(var_summary)
        elif var_type == "Categorical":
            var_types["categorical"].append(var_summary)
        elif var_type == "Text":
            var_types["text"].append(var_summary)
        elif var_type in ["DateTime", "Date", "Time"]:
            var_types["datetime"].append(var_summary)
        else:
            var_types["other"].append(var_summary)
    
    # Sort by importance (missing_pct, n_distinct, etc.)
    for var_type in var_types:
        var_types[var_type] = sorted(var_types[var_type], 
                                   key=lambda x: (x["missing_pct"], -x["n_distinct"]))
    
    # Tạo summary statistics (KHÔNG CÓ TARGET DETECTION)
    summary_stats = {
        "total_variables": len(variables),
        "by_type": {k: len(v) for k, v in var_types.items() if v},
        "missing_data": {
            "variables_with_missing": sum(1 for v in variables.values() if v.get("p_missing", 0) > 0),
            "avg_missing_pct": round(sum(v.get("p_missing", 0) for v in variables.values()) / len(variables), 3)
        },
        "data_issues": {
            "id_like_features": [],
            "high_cardinality_features": [],
            "highly_imbalanced_features": []
        }
    }
    
    # Detect data issues (KHÔNG DETECT TARGET)
    for var_name, var_info in variables.items():
        n_distinct = var_info.get("n_distinct", 0)
        var_type = var_info.get("type", "")
        
        # ID-like features
        total_rows = max(v.get("n", 1) for v in variables.values())
        if n_distinct > total_rows * 0.8:
            summary_stats["data_issues"]["id_like_features"].append(var_name)
        
        # High cardinality
        if n_distinct > 100:
            summary_stats["data_issues"]["high_cardinality_features"].append(var_name)
        
        # Highly imbalanced categorical features
        if var_type == "Categorical" and var_info.get("imbalance", 0) > 0.9:
            summary_stats["data_issues"]["highly_imbalanced_features"].append({
                "name": var_name,
                "imbalance": round(var_info.get("imbalance", 0), 3)
            })
    
    return {
        "summary_stats": summary_stats,
        "variables_by_type": var_types
    }

def create_enhanced_guideline_prompt(guideline_input: Dict) -> str:
    """
    Tạo prompt ENHANCED với variables summary (LLM tự xác định target)
    """
    
    task_info = guideline_input['task_info']
    summaries = guideline_input['summaries']
    profiles = guideline_input['profiles']
    
    # Extract key information
    dataset_name = task_info['name']
    task_desc = task_info['task_description']
    input_desc = task_info['input_description']
    output_desc = task_info['output_description']
    
    # Get sample data info
    sample_summary = list(summaries.values())[0] if summaries else {}
    n_rows = sample_summary.get('n_rows', 0)
    n_cols = sample_summary.get('n_cols', 0)
    
    # Get enhanced variables info
    sample_profile = list(profiles.values())[0] if profiles else {}
    alerts = sample_profile.get('alerts', [])
    variables = sample_profile.get('variables', {})
    
    # Create smart variables summary
    variables_summary = create_variables_summary(variables)
    
    prompt = f"""You are an expert Machine Learning engineer. Create a comprehensive guideline for an AutoML pipeline with 3 modules: Preprocessing, Modeling, and Evaluation.

## Dataset Information:
- **Dataset**: {dataset_name}
- **Task**: {task_desc}
- **Input**: {input_desc}
- **Output**: {output_desc}
- **Size**: {n_rows:,} rows, {n_cols} columns

## Data Quality Analysis:
- **Quality Issues**: {len(alerts)} alerts detected
- **Key Issues**: {', '.join(alerts[:3]) if alerts else 'None'}
"""
    
    # Add variables summary
    if variables_summary:
        stats = variables_summary["summary_stats"]
        vars_by_type = variables_summary["variables_by_type"]
        
        prompt += f"""
## Variables Analysis:
### Summary Statistics:
- **Total Variables**: {stats['total_variables']}
- **By Type**: {', '.join(f"{k}: {v}" for k, v in stats['by_type'].items())}
- **Missing Data**: {stats['missing_data']['variables_with_missing']} variables have missing values (avg: {stats['missing_data']['avg_missing_pct']:.1%})
"""
        
        # Add categorical variables
        if vars_by_type['categorical']:
            prompt += f"\n### Categorical Variables ({len(vars_by_type['categorical'])}):  \n"
            for var in vars_by_type['categorical'][:8]:  # More variables for LLM to analyze
                prompt += f"- **{var['name']}**: {var['n_distinct']} unique values, {var['missing_pct']:.1%} missing"
                if var.get('imbalance'):
                    prompt += f", imbalance: {var['imbalance']:.2f}"
                if var.get('top_values'):
                    top_items = list(var['top_values'].items())[:3]
                    prompt += f", top values: {top_items}"
                prompt += "\n"
        
        # Add numerical variables  
        if vars_by_type['numerical']:
            prompt += f"\n### Numerical Variables ({len(vars_by_type['numerical'])}):  \n"
            for var in vars_by_type['numerical'][:8]:  # More variables for LLM to analyze
                range_str = f"[{var.get('min', 'N/A')} - {var.get('max', 'N/A')}]"
                mean_str = f"mean: {var.get('mean', 'N/A')}"
                prompt += f"- **{var['name']}**: range {range_str}, {mean_str}, {var['missing_pct']:.1%} missing\n"
        
        # Add text variables
        if vars_by_type['text']:
            prompt += f"\n### Text Variables ({len(vars_by_type['text'])}):  \n"
            for var in vars_by_type['text']:
                prompt += f"- **{var['name']}**: {var['missing_pct']:.1%} missing\n"
        
        # Add datetime variables
        if vars_by_type['datetime']:
            prompt += f"\n### DateTime Variables ({len(vars_by_type['datetime'])}):  \n"
            for var in vars_by_type['datetime']:
                prompt += f"- **{var['name']}**: {var['missing_pct']:.1%} missing\n"
        
        # Add data issues
        issues_data = stats['data_issues']
        if any(issues_data.values()):
            prompt += f"\n### Data Issues:\n"
            if issues_data['id_like_features']:
                prompt += f"- **ID-like features**: {issues_data['id_like_features'][:3]}\n"
            if issues_data['high_cardinality_features']:
                prompt += f"- **High cardinality**: {issues_data['high_cardinality_features'][:3]}\n"
            if issues_data['highly_imbalanced_features']:
                imbalanced_names = [f['name'] for f in issues_data['highly_imbalanced_features'][:3]]
                prompt += f"- **Highly imbalanced**: {imbalanced_names}\n"
    
    prompt += """
## Required Guidelines:
Based on the task description and variable analysis, create detailed, actionable guidelines for each module. **You should identify the target variable(s) from the task description and variable characteristics.**

### 1. PREPROCESSING MODULE
- **Target Identification**: Identify the target variable(s) based on task description and variable analysis
- **Data Cleaning**: Specific steps based on detected issues and alerts
- **Feature Engineering**: Recommendations based on variable types and characteristics
- **Missing Value Handling**: Strategy based on missing patterns and variable importance
- **Feature Selection**: Methods appropriate for the data types and task
- **Data Splitting**: Strategy considering data size and task requirements

### 2. MODELING MODULE  
- **Algorithm Recommendations**: Specific algorithms based on task type, data size, and characteristics
- **Hyperparameter Tuning**: Efficient tuning strategies for the recommended algorithms
- **Model Selection**: Criteria and methods for selecting the best model
- **Cross-Validation**: Appropriate CV strategy for the dataset size and task

### 3. EVALUATION MODULE
- **Evaluation Metrics**: Specific metrics appropriate for the task and data characteristics
- **Validation Strategy**: Comprehensive evaluation approach
- **Performance Benchmarking**: Baseline comparisons and performance targets
- **Result Interpretation**: How to interpret results and identify model issues
Don't output your thinking, just output as the folowing format:
## Output Format: Your response must be the JSON format below:
Please provide your response in JSON format. For each sub-module in preprocessing, modeling, and evaluation, provide your recommendations along with a brief reasoning. **It is acceptable to provide an empty list or null for recommendations if none are suitable.**
```json
{
    "target_identification": {
        "target_variable": "identified_target_column_name",
        "reasoning": "explanation for target selection",
        "task_type": "classification/regression/etc"
    },
    "preprocessing": {
        "data_cleaning": ["specific step 1", "specific step 2"],
        "feature_engineering": ["specific technique 1", "specific technique 2"],
        "missing_values": ["strategy 1", "strategy 2"],
        "feature_selection": ["method 1", "method 2"],
        "data_splitting": {"train": 0.8, "val": 0.1, "test": 0.1, "strategy": "stratified"}
    },
    "modeling": {
        "recommended_algorithms": ["algorithm 1", "algorithm 2"],
        "hyperparameter_tuning": ["method 1", "method 2"],
        "model_selection": ["criteria 1", "criteria 2"],
        "cross_validation": {"method": "stratified_kfold", "folds": 5, "scoring": "appropriate_metric"}
    },
    "evaluation": {
        "metrics": ["metric 1", "metric 2"],
        "validation_strategy": ["approach 1", "approach 2"],
        "performance_benchmarking": ["baseline 1", "baseline 2"],
        "result_interpretation": ["interpretation 1", "interpretation 2"]
    }
}
```

Be specific and actionable in your recommendations. Identify the target variable based on the task description and variable analysis.
"""
    
    return prompt

def call_gemini_for_guideline(prompt: str, model: str = "gemini-2.5-flash") -> tuple[Optional[str], int, int]:
    """
    Gọi Gemini để sinh guideline
    Returns: (response, input_tokens, output_tokens)
    """
    
    # Count input tokens
    input_tokens = count_tokens_gemini(prompt)
    
    try:
        # 🔧 IMPROVED: Try multiple sources for API key
        api_key = None
        
        # Try from environment variable first
        api_key = os.getenv("GEMINI_API_KEY")
        
        # Try alternative environment variable names
        if not api_key:
            api_key = os.getenv("GOOGLE_API_KEY")
        
        # Try reading from .env file directly (backup)
        if not api_key:
            env_file = Path(".env")
            if env_file.exists():
                env_content = env_file.read_text()
                for line in env_content.split('\n'):
                    if line.startswith('GEMINI_API_KEY='):
                        api_key = line.split('=', 1)[1].strip().strip('"').strip("'")
                        break
                    elif line.startswith('GOOGLE_API_KEY='):
                        api_key = line.split('=', 1)[1].strip().strip('"').strip("'")
                        break
        
        if not api_key:
            print("  Gemini API key not found. Please check:")
            print("     1. GEMINI_API_KEY in .env file")
            print("     2. GEMINI_API_KEY environment variable")
            print("     3. GOOGLE_API_KEY environment variable")
            return None, input_tokens, 0
        
        # Configure Gemini
        genai.configure(api_key=api_key)
        print(f"   API key loaded successfully (ending with: ...{api_key[-4:]})")
        
        # Create model
        model_instance = genai.GenerativeModel(
            model_name=model,
            generation_config={
                "temperature": 0.7,
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 4000,
            }
        )
        
        # Generate response
        response = model_instance.generate_content(prompt)
        
        if response.text:
            output_tokens = count_tokens_gemini(response.text)
            return response.text, input_tokens, output_tokens
        else:
            print(f"  No response text from Gemini")
            return None, input_tokens, 0
    
    except Exception as e:
        print(f" Error calling Gemini: {e}")
        return None, input_tokens, 0

def generate_guidelines_for_dataset(guideline_input: Dict, output_dir: Path) -> Optional[Dict]:
    """
    Sinh guideline cho 1 dataset
    """
    
    dataset_name = guideline_input['task_info']['name']
    dataset_id = guideline_input['task_info']['dataset_id']
    
    print(f" Generating guidelines for: {dataset_name}")
    
    # Tạo enhanced prompt
    prompt = create_enhanced_guideline_prompt(guideline_input)
    
    # Gọi Gemini
    gemini_response, input_tokens, output_tokens = call_gemini_for_guideline(prompt, model="gemini-2.5-flash")
    
    # Print token usage
    print(f"  Token Usage:")
    print(f"     Input: {input_tokens:,} tokens")
    print(f"     Output: {output_tokens:,} tokens")
    print(f"     Total: {input_tokens + output_tokens:,} tokens")
    
    # Estimate cost for Gemini 2.5 Flash (approximate pricing)
    input_cost = input_tokens * 0.00015 / 1000  
    output_cost = output_tokens * 0.0006 / 1000
    total_cost = input_cost + output_cost
    print(f"     Estimated Cost: ${total_cost:.6f}")
    
    if gemini_response:
        # Parse JSON response
        try:
            # Clean response if needed
            response_clean = gemini_response.strip()
            if response_clean.startswith("```json"):
                response_clean = response_clean[7:]
            if response_clean.endswith("```"):
                response_clean = response_clean[:-3]
            
            guidelines = json.loads(response_clean)
            print(f"    Guidelines parsed successfully")
            
            # Print target identification
            if "target_identification" in guidelines:
                target_info = guidelines["target_identification"]
                print(f"   Target identified: {target_info.get('target_variable', 'N/A')}")
                print(f"     Task type: {target_info.get('task_type', 'N/A')}")
            
        except json.JSONDecodeError as e:
            print(f"   JSON parse error: {e}")
            guidelines = {
                "raw_response": gemini_response,
                "parse_error": str(e)
            }
        
        # Thêm metadata
        result = {
            "dataset_info": {
                "id": dataset_id,
                "name": dataset_name,
                "generated_at": datetime.now().isoformat(),
                "model_used": "gemini-2.5-flash",
                "token_usage": {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens,
                    "estimated_cost": total_cost
                }
            },
            "guidelines": guidelines
        }
        
        # Save individual guideline
        safe_name = dataset_name.replace(' ', '_').replace('/', '_')
        output_file = output_dir / f"{dataset_id}_{safe_name}_guideline.json"
        output_file.write_text(json.dumps(result, indent=2, ensure_ascii=False))
        
        print(f"  Guidelines saved to: {output_file.name}")
        return result
    
    else:
        print(f"   Failed to generate guidelines for {dataset_name}")
        return None

def generate_all_guidelines(guideline_inputs: List[Dict], output_dir: str = "guidelines_output") -> List[Dict]:
    """
    Sinh guidelines cho tất cả datasets
    """
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    all_guidelines = []
    total_input_tokens = 0
    total_output_tokens = 0
    total_cost = 0
    
    print(f" Generating guidelines for {len(guideline_inputs)} datasets using Gemini 2.5 Flash...")
    print("=" * 70)
    
    for i, guideline_input in enumerate(guideline_inputs, 1):
        print(f"\n Dataset {i}/{len(guideline_inputs)}")
        print("-" * 40)
        
        guideline_result = generate_guidelines_for_dataset(guideline_input, output_path)
        
        if guideline_result:
            all_guidelines.append(guideline_result)
            
            # Accumulate totals
            token_usage = guideline_result["dataset_info"]["token_usage"]
            total_input_tokens += token_usage["input_tokens"]
            total_output_tokens += token_usage["output_tokens"]
            total_cost += token_usage["estimated_cost"]
    
    # Save consolidated guidelines
    consolidated_file = output_path / "all_guidelines.json"
    consolidated_file.write_text(json.dumps(all_guidelines, indent=2, ensure_ascii=False))
    
    # Print summary
    print(f"\n" + "=" * 70)
    print(f" All guidelines generated!")
    print(f" Individual files: {output_path}")
    print(f" Consolidated file: {consolidated_file}")
    print(f" Total datasets processed: {len(all_guidelines)}")
    print(f"\n TOTAL TOKEN USAGE (Gemini 2.5 Flash):")
    print(f"   Input tokens: {total_input_tokens:,}")
    print(f"   Output tokens: {total_output_tokens:,}")
    print(f"   Total tokens: {total_input_tokens + total_output_tokens:,}")
    print(f"   Total estimated cost: ${total_cost:.6f}")
    
    return all_guidelines

def prepare_for_guideline_generation(meta_file: str = 'meta-data.json', 
                                   profiling_root: str = 'profiling_results', 
                                   output_file: str = 'guideline_input.json'):
    """
    Chuẩn bị input cho guideline generation
    """
    meta_data = json.loads(Path(meta_file).read_text(encoding='utf-8'))
    profiling_dir = Path(profiling_root)
    guideline_inputs = []
    
    print(f" Extracting inputs for {len(meta_data)} datasets...")
    
    for dataset in meta_data:
        ds_id = str(dataset.get('id'))
        ds_name = dataset.get('name', f'dataset_{ds_id}')
        safe_name = ds_name.replace(' ', '_').replace('/', '_')
        
        dataset_dir = profiling_dir / f'{ds_id}_{safe_name}'
        
        if dataset_dir.exists():
            print(f" Processing {ds_name}...")
            guideline_input = extract_guideline_input(dataset_dir, dataset)
            guideline_inputs.append(guideline_input)
            print(f"   Done")
        else:
            print(f"  Skip {ds_name} - not found")
    
    # Save results
    output_path = Path(output_file)
    output_path.write_text(json.dumps(guideline_inputs, indent=2, ensure_ascii=False))
    
    print(f"\n Saved to: {output_path}")
    print(f" Total: {len(guideline_inputs)} datasets")
    
    return guideline_inputs

def main():
    """
    Main function: Chạy toàn bộ pipeline
    """
    
    print(" AutoML Guideline Generation Pipeline (Gemini 2.5 Flash)")
    print("=" * 50)
    
    # Step 1: Prepare guideline inputs
    print("\n Step 1: Preparing guideline inputs...")
    guideline_inputs = prepare_for_guideline_generation()
    
    if not guideline_inputs:
        print(" No guideline inputs prepared. Exiting...")
        return
    
    # Step 2: Generate guidelines using Gemini
    print("\n Step 2: Generating guidelines using Gemini 2.5 Flash...")
    guidelines = generate_all_guidelines(guideline_inputs)
    
    if guidelines:
        print(f"\n Pipeline completed successfully!")
        
        # Save summary
        summary = {
            "generation_summary": {
                "total_datasets": len(guidelines),
                "generated_at": datetime.now().isoformat(),
                "model_used": "gemini-2.5-flash",
                "approach": "simplified_without_target_detection"
            },
            "token_usage_summary": {
                "total_input_tokens": sum(g["dataset_info"]["token_usage"]["input_tokens"] for g in guidelines),
                "total_output_tokens": sum(g["dataset_info"]["token_usage"]["output_tokens"] for g in guidelines),
                "total_cost": sum(g["dataset_info"]["token_usage"]["estimated_cost"] for g in guidelines)
            }
        }
        
        Path("guidelines_output/generation_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False)
        )
        
    else:
        print(" No guidelines generated successfully!")

if __name__ == "__main__":
    main()
    