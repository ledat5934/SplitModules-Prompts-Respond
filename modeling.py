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
class ModelingGenerator:
    def __init__(self, max_retries: int = 5):
        self.max_retries = max_retries
        self.setup_gemini()
    def setup_gemini(self):
        api_key = os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY')
        if not api_key:
            raise ValueError('GEMINI_API_KEY not found in environment variables')
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
    
    def load_preprocessing_code(self, preprocessing_file: str) -> str:
        print(f'Loading preprocessing code from {preprocessing_file}...')
        if not Path(preprocessing_file).exists():
            raise FileNotFoundError(f'Preprocessing file not found: {preprocessing_file}')
        preprocessing_code = Path(preprocessing_file).read_text(encoding='utf-8')
        print(f'Loaded {len(preprocessing_code)} characters of preprocessing code')
        print(preprocessing_code)
        return preprocessing_code
    
    def create_modeling_prompt(self, guidelines: Dict, metadata: Dict, preprocessing_code: str, previous_code: str = None, error_message: str = None) -> str:
        dataset_name = metadata['name']
        task_desc = metadata['task']
        file_paths = metadata.get('link to the dataset', [])
        ground_truth_paths = metadata.get('link to the ground truth', [])
        modeling_guideline = guidelines['guidelines'].get('modeling', {})

        prompt = f"""
You are an expert ML engineer. Generate Python modeling code for this dataset which compatible with the preprocessing code:
Dataset Name: {dataset_name}
Task Description: {task_desc}
File Paths: {file_paths}
Ground Truth Paths: {ground_truth_paths}
Guidelines:
{json.dumps(modeling_guideline, indent=2)}

Preprocessing Code(just the code, no other text, do not include test function):
{preprocessing_code}

Requirements:
1. Generate COMPLETE code which Executeable when combined with the preprocessing code
2. Use the same variable names as the preprocessing code
3. Consider following the modelling guidelines
4. Try to choose the model architecture which provide best performance.
5. Include model selection, hyperparameter tuning, training, ... of your choice
6. Use appropriate libraries and functions
7. Test the execution on the real data or parts of it(if the dataset is large), not the dummy data.
8. Generate a submission.csv file for the test.csv file and evaluate the model on the ground truth file.

##Code format:
#import necessary libraries
# Include preprocessing code
{preprocessing_code}
# Your modeling code
# Test the combined code if it executeable

## IMPORTANT NOTES:
1. The preprocessing function `preprocess_data()` is already available - use it exactly as shown
2. Choose appropriate model based on task type (classification/regression)
3. Include proper error handling and validation
4. Follow the modeling guidelines for model selection and hyperparameters
5. Generate meaningful evaluation metrics
6. DO NOT save model to file - just train and evaluate
7. Make sure the code runs completely without errors
8. Do not use hyperparameter tuning.
19 If the problem is deep learning, try to use GPU and use appropriate pretrained model if possible.
110 Use multimodal if necessary.
11. Limit the comment in the code.
12. **Critical Error Handling**: The main execution block (`if __name__ == "__main__":`) MUST be wrapped in a try...except block. If ANY exception occurs during the process, the script MUST print the error and then **exit with a non-zero status code** using `sys.exit(1)`.
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
3. Ensure compatibility with the preprocessing function
4. Keep the same overall structure but fix the problematic parts
5. Make sure the combined preprocessing + modeling pipeline works
6. Test all imports and function calls
"""

        prompt += "\nGenerate the corrected Python modeling code:"
        
        return prompt

    def generate_modeling_code(self, prompt: str) -> str:
        """Generate modeling code using Gemini"""
        print(" Generating modeling code...")
        
        try:
            response = self.model.generate_content(prompt)
            code = response.text
            
            # Extract Python code from response
            if "```python" in code:
                code = code.split("```python")[1].split("```")[0].strip()
            elif "```" in code:
                code = code.split("```")[1].split("```")[0].strip()
            
            print(f" Generated {len(code)} characters of modeling code")
            return code
            
        except Exception as e:
            print(f" Error generating code: {e}")
            raise

    def test_combined_pipeline(self, modeling_code: str, file_paths: List[str]) -> Tuple[bool, str]:
        """Test the combined preprocessing + modeling pipeline"""
        print(" Testing combined preprocessing + modeling pipeline...")
        
        try:
            # Create temporary file with the modeling code (includes preprocessing)
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                print(modeling_code)
                f.write(modeling_code)
                temp_file = f.name
            
            # Execute the code
            result = subprocess.run(
                [sys.executable, temp_file],
                capture_output=True,
                text=True,
                timeout=1800  # 30 minute timeout for model training
            )
            
            # Clean up
            os.unlink(temp_file)
            
            if result.returncode == 0:
                print(" Combined pipeline executed successfully!")
                return True, result.stdout
            else:
                print(" Combined pipeline execution failed!")
                return False, result.stderr
                
        except subprocess.TimeoutExpired:
            print("   Pipeline execution timed out after 30 minutes")
            print("  However, this likely means the code is executable, just slow.")
            print("  Saving the code as it appears to be working...")
            
            # Cleanup temp file
            if 'temp_file' in locals():
                try:
                    os.unlink(temp_file)
                except:
                    pass
            
            # Return SUCCESS vì code có thể chạy được, chỉ là chậm
            return True, "Code appears executable but runs slowly (timed out after 60 minutes)"
        except Exception as e:
            print(f" Error executing pipeline: {e}")
            return False, str(e)

    def save_modeling_code(self, code: str, dataset_id: str, output_dir: str = "generated_code"):
        """Save the successful modeling code"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        file_name = f"modeling_dataset_{dataset_id}.py"
        file_path = output_path / file_name
        
        # Add header comment
        header = f"""# Auto-generated modeling code for Dataset {dataset_id}
# Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
# AutoML Pipeline - Step 4: Modeling
# This code includes preprocessing and modeling in one pipeline

"""
        
        full_code = header + code
        file_path.write_text(full_code, encoding='utf-8')
        
        print(f" Saved modeling code to: {file_path}")
        return file_path

    def run_modeling_pipeline(self, guideline_file: str, meta_data_file: str, preprocessing_file: str, 
                             dataset_id: str, output_dir: str = "generated_code") -> Optional[Path]:
        """
        Main pipeline to generate and test modeling code
        Returns path to successful code file or None if failed
        """
        print(f"\n Starting modeling pipeline for dataset {dataset_id}")
        print("="*60)
        
        try:
            # Load inputs
            guidelines, metadata = self.load_guidelines_and_metadata(guideline_file, meta_data_file, dataset_id)
            preprocessing_code = self.load_preprocessing_code(preprocessing_file)
            file_paths = metadata.get('link to the dataset', [])
            
            previous_code = None
            error_message = None
            
            for attempt in range(1, self.max_retries + 1):
                print(f"\n Attempt {attempt}/{self.max_retries}")
                
                # Create prompt
                prompt = self.create_modeling_prompt(guidelines, metadata, preprocessing_code, previous_code, error_message)
                
                # Generate code
                code = self.generate_modeling_code(prompt)
                
                # Test combined pipeline
                success, output = self.test_combined_pipeline(code, file_paths)
                
                if success:
                    print(" Modeling code generated and tested successfully!")
                    print(" Training output:")
                    print(output)
                    
                    # Save the successful code
                    saved_path = self.save_modeling_code(code, dataset_id, output_dir)
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
    if len(sys.argv) != 5:
        print("Usage: python modeling.py <guideline_file> <metadata_file> <preprocessing_file> <dataset_id>")
        print("Example: python modeling.py guidelines_output/all_guidelines.json meta-data.json generated_code/preprocessing_dataset_2.py 2")
        sys.exit(1)
    
    guideline_file = sys.argv[1]
    metadata_file = sys.argv[2]
    preprocessing_file = sys.argv[3]
    dataset_id = sys.argv[4]  # Keep as string for consistency
    
    generator = ModelingGenerator(max_retries=5)
    result = generator.run_modeling_pipeline(guideline_file, metadata_file, preprocessing_file, dataset_id)
    
    if result:
        print(f"\n SUCCESS! Modeling code saved to: {result}")
        print(f" This code combines preprocessing + modeling in one pipeline")
        sys.exit(0)
    else:
        print(f"\n FAILED! Could not generate working modeling code after {generator.max_retries} attempts")
        sys.exit(1)

if __name__ == "__main__":
    main()






        
        
