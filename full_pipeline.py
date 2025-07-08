import sys
import json 
import subprocess
from pathlib import Path
from datetime import datetime

from profile_data import run_profiling
from guideline_create import prepare_for_guideline_generation, generate_all_guidelines
from preprocessing import PreprocessingGenerator
from modeling import ModelingGenerator

def run_single_dataset_profiling(dataset_id: str, meta_file: str = 'meta-data.json', output_root: str = 'profiling_results'):
    print(f'1: Profiling dataset {dataset_id}')
    meta_path = Path(meta_file)
    if not meta_path.exists():
        print(f'meta-data.json not found in {meta_path}')
        return False
    
    all_metadata = json.loads(meta_path.read_text(encoding='utf-8'))
    target_dataset = None
    for dataset in all_metadata:
        if str(dataset.get('id')) == str(dataset_id):
            target_dataset = dataset
            break

    if not target_dataset:
        print(f'Dataset {dataset_id} not found in {meta_file}')
        return False
    
    temp_meta = [target_dataset]
    temp_meta_file = f'temp_meta_{dataset_id}.json'
    try:
        Path(temp_meta_file).write_text(json.dumps(temp_meta, indent=2, ensure_ascii=False), encoding='utf-8')
        run_profiling(temp_meta_file, output_root)
        print(f'Profiling results saved to {output_root}')
        return True
    except Exception as e:
        print(f'Error during profiling: {e}')
        if Path(temp_meta_file).exists():
            Path(temp_meta_file).unlink()
        return False
    
def run_single_dataset_guidelines(dataset_id: str):
    """Generate guidelines chỉ cho một dataset cụ thể"""
    print(f"\n STEP 2: GENERATING GUIDELINES FOR DATASET {dataset_id}")
    print("="*60)
    
    try:
        # Chuẩn bị guideline inputs (sẽ tạo cho tất cả nhưng ta chỉ cần dataset này)
        guideline_inputs = prepare_for_guideline_generation()
        
        # Lọc chỉ lấy dataset cần thiết
        target_input = None
        for inp in guideline_inputs:
            if str(inp['task_info']['dataset_id']) == str(dataset_id):
                target_input = inp
                break
        
        if not target_input:
            print(f" Không tìm thấy guideline input cho dataset {dataset_id}")
            return False
        
        # Generate guidelines chỉ cho dataset này
        result = generate_all_guidelines([target_input])
        
        if result:
            print(f" Hoàn tất tạo guidelines cho dataset {dataset_id}")
            return True
        else:
            print(f" Không thể tạo guidelines cho dataset {dataset_id}")
            return False
            
    except Exception as e:
        print(f" Lỗi khi tạo guidelines: {e}")
        return False

def run_single_dataset_preprocessing(dataset_id: str):
    """Generate preprocessing code cho một dataset cụ thể"""
    print(f"\n STEP 3: GENERATING PREPROCESSING CODE FOR DATASET {dataset_id}")
    print("="*60)
    
    try:
        # Sử dụng class PreprocessingGenerator từ module có sẵn
        generator = PreprocessingGenerator(max_retries=5)
        
        # Gọi pipeline với các file cần thiết
        result = generator.run_preprocessing_pipeline(
            guideline_file="guidelines_output/all_guidelines.json",
            meta_data_file="meta-data.json", 
            dataset_id=dataset_id
        )
        
        if result:
            print(f" Hoàn tất tạo preprocessing code cho dataset {dataset_id}")
            print(f"   File được lưu tại: {result}")
            return True
        else:
            print(f" Không thể tạo preprocessing code cho dataset {dataset_id}")
            return False
            
    except Exception as e:
        print(f" Lỗi khi tạo preprocessing code: {e}")
        return False

def run_single_dataset_modeling(dataset_id: str):
    """Generate modeling code cho một dataset cụ thể"""
    print(f"\n STEP 4: GENERATING MODELING CODE FOR DATASET {dataset_id}")
    print("="*60)
    
    try:
        # Tìm preprocessing file đã được tạo
        preprocessing_file = Path("generated_code") / f"preprocessing_dataset_{dataset_id}.py"
        
        if not preprocessing_file.exists():
            print(f" Không tìm thấy preprocessing file: {preprocessing_file}")
            return False
        
        # Sử dụng class ModelingGenerator từ module có sẵn
        generator = ModelingGenerator(max_retries=5)
        
        # Gọi pipeline với các file cần thiết
        result = generator.run_modeling_pipeline(
            guideline_file="guidelines_output/all_guidelines.json",
            meta_data_file="meta-data.json",
            preprocessing_file=str(preprocessing_file),
            dataset_id=dataset_id
        )
        
        if result:
            print(f" Hoàn tất tạo modeling code cho dataset {dataset_id}")
            print(f"   File được lưu tại: {result}")
            return True
        else:
            print(f" Không thể tạo modeling code cho dataset {dataset_id}")
            return False
            
    except Exception as e:
        print(f" Lỗi khi tạo modeling code: {e}")
        return False

def run_full_pipeline(dataset_id: str):
    """Chạy toàn bộ pipeline cho một dataset"""
    print(f"\n STARTING FULL AUTOML PIPELINE FOR DATASET {dataset_id}")
    print("="*80)
    
    start_time = datetime.now()
    
    try:
        # Kiểm tra requirements
        required_files = ["meta-data.json", ".env"]
        missing_files = [f for f in required_files if not Path(f).exists()]
        
        if missing_files:
            print(f" Thiếu các file cần thiết: {missing_files}")
            print("Vui lòng đảm bảo có file meta-data.json và .env với GEMINI_API_KEY")
            return False
        
        # Step 1: Profiling
        if not run_single_dataset_profiling(dataset_id):
            print("  PIPELINE FAILED at Step 1: Profiling")
            return False
        
        # Step 2: Guidelines
        if not run_single_dataset_guidelines(dataset_id):
            print(" PIPELINE FAILED at Step 2: Guidelines Generation")
            return False
        
        # Step 3: Preprocessing
        if not run_single_dataset_preprocessing(dataset_id):
            print(" PIPELINE FAILED at Step 3: Preprocessing Code Generation")
            return False
        
        # Step 4: Modeling
        if not run_single_dataset_modeling(dataset_id):
            print(" PIPELINE FAILED at Step 4: Modeling Code Generation")
            return False
        
        # Success!
        end_time = datetime.now()
        duration = end_time - start_time
        
        print(f"\n PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f" Total time: {duration}")
        print(f" Generated files:")
        print(f"   - Profiling: profiling_results/{dataset_id}_*/")
        print(f"   - Guidelines: guidelines_output/")
        print(f"   - Preprocessing: generated_code/preprocessing_dataset_{dataset_id}.py")
        print(f"   - Modeling: generated_code/modeling_dataset_{dataset_id}.py")
        print("\n Your AutoML pipeline is ready!")
        print(f"   To run the complete model, execute:")
        print(f"   python generated_code/modeling_dataset_{dataset_id}.py")
        
        return True
        
    except Exception as e:
        print(f"\n PIPELINE FAILED with exception: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main execution function"""
    if len(sys.argv) != 2:
        print("Usage: python full_pipeline.py <dataset_id>")
        print("Example: python full_pipeline.py 2")
        print("\nAvailable datasets in meta-data.json:")
        
        # Hiển thị danh sách datasets có sẵn
        try:
            meta_data = json.loads(Path("meta-data.json").read_text(encoding='utf-8'))
            for dataset in meta_data:
                print(f"  - ID: {dataset.get('id')}, Name: {dataset.get('name')}")
        except:
            print("  (Cannot read meta-data.json)")
        
        sys.exit(1)
    
    dataset_id = sys.argv[1]
    
    try:
        # Run full pipeline
        success = run_full_pipeline(dataset_id)
        
        if success:
            print(f"\n SUCCESS! Full pipeline completed for dataset {dataset_id}")
            sys.exit(0)
        else:
            print(f"\n FAILED! Pipeline could not complete for dataset {dataset_id}")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print(f"\n  Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()