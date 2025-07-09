import json
import sys
import traceback
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import google.generativeai as genai
import os
from datetime import datetime
from dotenv import load_dotenv
import tempfile
import pandas as pd

load_dotenv()

class PreprocessingGenerator:
    def __init__(self, max_retries: int = 5):
        self.max_retries = max_retries
        self.setup_gemini()
    def setup_gemini(self):
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(
            model_name="gemini-2.5-flash",
            generation_config={
                "temperature": 0,  # Lower temperature for more consistent code
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 16000,
            }
        )
        print('Gemini API setup complete')

    def load_guidelines_and_metadata(self, guideline_file: str,meta_data_file: str, dataset_id: str) -> Tuple[Dict, Dict]:
        print(f'Loading guidelines and metadata for {dataset_id}...')
        if not Path(guideline_file).exists():
            raise FileNotFoundError(f'Guideline file not found: {guideline_file}')
        
        guidelines_data = json.loads(Path(guideline_file).read_text(encoding='utf-8'))
        dataset_guidelines = None
        for guideline in guidelines_data:
            if str(guideline["dataset_info"]["id"]) == str(dataset_id):
                dataset_guidelines = guideline
                break

        if not dataset_guidelines:
            raise ValueError(f'No guidelines found for dataset {dataset_id}')
        
        meta_data = json.loads(Path(meta_data_file).read_text(encoding='utf-8'))
        if not meta_data:
            raise ValueError(f'No metadata found for dataset {dataset_id}')
        dataset_metadata = None
        for meta in meta_data:
            if str(meta["id"]) == str(dataset_id):
                dataset_metadata = meta
                break
        if not dataset_metadata:
            raise ValueError(f'No metadata found for dataset {dataset_id}')
        
        print(f'Loaded {len(dataset_guidelines)} guidelines and {len(dataset_metadata)} metadata for {dataset_id}')
        print(dataset_guidelines)
        print(dataset_metadata)
        return dataset_guidelines, dataset_metadata
    
    def create_preprocessing_prompt(self, guidelines: Dict, metadata: Dict, previous_code: str = None, error_message: str = None) -> str:
        dataset_name = metadata['name']
        task_desc = metadata['task']
        file_paths = metadata.get('link to the dataset', [])
        input_desc = metadata.get('input_data', '')
        output_desc = metadata.get('output_data', '')
        data_file_desc = metadata.get('data file description', '')

        preprocessing_guideline = guidelines['guidelines'].get('preprocessing', {})
        target_info = guidelines["guidelines"].get("target_identification", {})

        prompt = f"""
You are a professional Machine Learning Engineer.
Generate complete and executable Python preprocessing code for the dataset below.

## DATASET INFO:
- Name: {dataset_name}
- Task: {task_desc}
- Input: {input_desc}
- Output: {output_desc}
- Data files: {data_file_desc}
- File paths: {file_paths}

## PREPROCESSING GUIDELINES:
{json.dumps(preprocessing_guideline, indent=2)}

## TARGET INFO:
{json.dumps(target_info, indent=2)}

## REQUIREMENTS:
1. Generate COMPLETE, EXECUTABLE Python code
2. Include all necessary imports
3. Handle file loading from the provided paths
4. Follow the preprocessing guidelines exactly
5. Return a function `preprocess_data()` that takes file paths and returns (X_train, X_test, y_train, y_test)
6. Include error handling and data validation
7. Use pandas, scikit-learn, numpy as main libraries
9. Limit the comment in the code.
10. The entire program(include preprocessing, modeling, evaluation) should run within 30 minutes with ML algorithm and 60 minutes with Deep learning algorithm, so do not over feature engineering the data, or you can use feature selection to reduce the number of features.
11. Test the execution on the real data or parts of it(if the dataset is large), not the dummy data.
12. **Critical Error Handling**: The main execution block (`if __name__ == "__main__":`) MUST be wrapped in a try...except block. If ANY exception occurs during the process, the script MUST print the error and then **exit with a non-zero status code** using `sys.exit(1)`.
## CODE STRUCTURE:
#import necessary libraries

def preprocess_data(file_paths):
    \"\"\"
    Preprocess data according to guidelines
    Returns: X_train, X_test, y_train, y_test
    \"\"\"
    # Your preprocessing code here
    return X_train, X_test, y_train, y_test

# Test the function
if __name__ == "__main__":
    file_paths = {file_paths}
    X_train, X_test, y_train, y_test = preprocess_data(file_paths)
    print(f"Preprocessing complete!")
    print(f"X_train shape: {{X_train.shape}}")
    print(f"X_test shape: {{X_test.shape}}")
    print(f"y_train shape: {{y_train.shape}}")
    print(f"y_test shape: {{y_test.shape}}")
```

"""

        # Add retry context if this is a retry
        if previous_code and error_message:
            prompt += f"""
## PREVIOUS ATTEMPT FAILED:
Previous code:
```python
{previous_code}
```

Error message:
{error_message}
## FIX INSTRUCTIONS:
1. Analyze the error carefully
2. Fix the specific issue that caused the error
3. Ensure the code runs without errors
4. Keep the same overall structure but fix the problematic parts
"""

        prompt += "\nGenerate the corrected Python code:"
        
        return prompt

    def generate_preprocessing_code(self, prompt: str) -> str:
        """Generate preprocessing code using Gemini"""
        print(" Generating preprocessing code...")
        
        try:
            response = self.model.generate_content(prompt)
            code = response.text
            
            # Extract Python code from response
            if "```python" in code:
                code = code.split("```python")[1].split("```")[0].strip()
            elif "```" in code:
                code = code.split("```")[1].split("```")[0].strip()
            
            print(f" Generated {len(code)} characters of code")
            return code
            
        except Exception as e:
            print(f" Error generating code: {e}")
            raise

    def execute_code(self, code: str, file_paths: List[str]) -> Tuple[bool, str]:
        """Execute preprocessing code and return success status and output/error"""
        print(" Executing preprocessing code...")
        
        try:
            # Create temporary file with the code
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                temp_file = f.name
            
            # Execute the code
            result = subprocess.run(
                [sys.executable, temp_file],
                capture_output=True,
                text=True,
                timeout=1200  # 20 minute timeout
            )
            
            # Clean up
            os.unlink(temp_file)
            
            if result.returncode == 0:
                print(" Code executed successfully!")
                return True, result.stdout
            else:
                print(" Code execution failed!")
                return False, result.stderr
                
        except subprocess.TimeoutExpired:
            print(" Code execution timed out!")
            return True, "Code execution timed out after 20 minutes"
        except Exception as e:
            print(f" Error executing code: {e}")
            return False, str(e)

    def save_preprocessing_code(self, code: str, dataset_id: str, output_dir: str = "generated_code"):
        """Save the successful preprocessing code"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        file_name = f"preprocessing_dataset_{dataset_id}.py"
        file_path = output_path / file_name
        
        # Add header comment
        header = f"""# Auto-generated preprocessing code for Dataset {dataset_id}
# Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
# AutoML Pipeline - Step 3: Preprocessing

"""
        
        full_code = header + code
        file_path.write_text(full_code, encoding='utf-8')
        
        print(f" Saved preprocessing code to: {file_path}")
        return file_path

    def run_preprocessing_pipeline(self, guideline_file: str, meta_data_file: str, dataset_id: str, output_dir: str = "generated_code") -> Optional[Path]:
        """
        Main pipeline to generate and test preprocessing code
        Returns path to successful code file or None if failed
        """
        print(f"\n Starting preprocessing pipeline for dataset {dataset_id}")
        print("="*60)
        
        try:
            # Load data
            guidelines, metadata = self.load_guidelines_and_metadata(guideline_file, meta_data_file, dataset_id)
            file_paths = metadata.get('link to the dataset', [])
            
            previous_code = None
            error_message = None
            
            for attempt in range(1, self.max_retries + 1):
                print(f"\n Attempt {attempt}/{self.max_retries}")
                
                # Create prompt
                prompt = self.create_preprocessing_prompt(guidelines, metadata, previous_code, error_message)
                
                # Generate code
                code = self.generate_preprocessing_code(prompt)
                
                # Execute code
                success, output = self.execute_code(code, file_paths)
                
                if success:
                    print(" Preprocessing code generated and tested successfully!")
                    # Save the successful code
                    saved_path = self.save_preprocessing_code(code, dataset_id, output_dir)
                    return saved_path
                else:
                    print(f" Attempt {attempt} failed")
                    print(f"Error: {output}")
                    
                    # Prepare for retry
                    previous_code = code
                    error_message = output
                    
                    if attempt < self.max_retries:
                        print(f" Retrying with error context...")
                    else:
                        print(f" All {self.max_retries} attempts failed!")
            
            return None
            
        except Exception as e:
            print(f" Pipeline failed with exception: {e}")
            traceback.print_exc()
            return None

def main():
    """Main execution function"""
    if len(sys.argv) != 4:
        print("Usage: python preprocessing.py <guideline_file> <metadata_file> <dataset_id>")
        sys.exit(1)
    
    guideline_file = sys.argv[1]
    metadata_file = sys.argv[2]
    dataset_id = int(sys.argv[3])
    
    generator = PreprocessingGenerator(max_retries=5)
    result = generator.run_preprocessing_pipeline(guideline_file, metadata_file, dataset_id)
    
    if result:
        print(f"\n SUCCESS! Preprocessing code saved to: {result}")
        sys.exit(0)
    else:
        print(f"\n FAILED! Could not generate working preprocessing code after {generator.max_retries} attempts")
        sys.exit(1)

if __name__ == "__main__":
    main()
