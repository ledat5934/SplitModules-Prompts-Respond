import sys
import json
import subprocess
import argparse
import traceback
import os
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# Third-party imports (ensure these are installed)
from dotenv import load_dotenv

# ==============================================================================
# CONFIGURATION & SETUP
# ==============================================================================
load_dotenv()

# --- Logger Setup ---
import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ==============================================================================
# PIPELINE STEPS (Importing and wrapping existing logic)
# ==============================================================================
# NOTE: The following functions assume the original scripts (or their logic) are available.

# Bước 0: Tạo metadata (hàm đã được sửa đổi để import)
from metadata_create import generate_metadata_for_path
# Bước 1: Phân tích dữ liệu
from profile_data import run_profiling
# Bước 2: Tạo hướng dẫn
from guideline_create import prepare_for_guideline_generation, generate_all_guidelines
# Bước 3: Tạo code tiền xử lý
from preprocessing import PreprocessingGenerator
# Bước 4: Tạo code mô hình hóa
from modeling import ModelingGenerator
# Bước 5: Lắp ráp code cuối cùng
from code_assembler import CodeAssembler


def run_step_1_profiling(dataset_id: str, meta_file: str, output_root: str) -> bool:
    """Wrapper for the data profiling step."""
    logger.info(f"STEP 1: Profiling dataset {dataset_id}")
    try:
        # Create a temporary metadata file containing only the target dataset
        all_metadata = json.loads(Path(meta_file).read_text(encoding='utf-8'))
        target_dataset = next((ds for ds in all_metadata if str(ds.get('id')) == dataset_id), None)
        if not target_dataset:
            logger.error(f"Dataset {dataset_id} not found in {meta_file}")
            return False
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
            json.dump([target_dataset], f, indent=2, ensure_ascii=False)
            temp_meta_file = f.name
        
        # Giả sử run_profiling có thể nhận file meta tạm thời
        run_profiling(temp_meta_file, output_root)
        os.unlink(temp_meta_file) # Clean up
        logger.info(f"Profiling successful for dataset {dataset_id}")
        return True
    except Exception as e:
        logger.error(f"Error during profiling: {e}", exc_info=True)
        return False

def run_step_2_guidelines(dataset_id: str) -> bool:
    """Wrapper for the guideline generation step."""
    logger.info(f"STEP 2: Generating Guidelines for dataset {dataset_id}")
    try:
        all_inputs = prepare_for_guideline_generation()
        target_input = next((inp for inp in all_inputs if str(inp.get('task_info', {}).get('dataset_id')) == dataset_id), None)
        if not target_input:
            logger.error(f"Could not prepare guideline input for dataset {dataset_id}")
            return False
        
        result = generate_all_guidelines([target_input])
        return bool(result)
    except Exception as e:
        logger.error(f"Error during guideline generation: {e}", exc_info=True)
        return False

def run_step_3_preprocessing(dataset_id: str) -> bool:
    """Wrapper for the preprocessing code generation step."""
    logger.info(f"STEP 3: Generating Preprocessing Code for dataset {dataset_id}")
    try:
        generator = PreprocessingGenerator()
        result_path = generator.run_preprocessing_pipeline(
            guideline_file="guidelines_output/all_guidelines.json",
            meta_data_file="meta-data.json",
            dataset_id=dataset_id
        )
        return bool(result_path)
    except Exception as e:
        logger.error(f"Error during preprocessing code generation: {e}", exc_info=True)
        return False

def run_step_4_modeling(dataset_id: str) -> bool:
    """Wrapper for the modeling code generation step."""
    logger.info(f"STEP 4: Generating Modeling Code for dataset {dataset_id}")
    try:
        preprocessing_file = Path("generated_code") / f"preprocessing_dataset_{dataset_id}.py"
        if not preprocessing_file.exists():
            logger.error(f"Required preprocessing file not found: {preprocessing_file}")
            return False
            
        generator = ModelingGenerator()
        result_path = generator.run_modeling_pipeline(
            guideline_file="guidelines_output/all_guidelines.json",
            meta_data_file="meta-data.json",
            preprocessing_file=str(preprocessing_file),
            dataset_id=dataset_id
        )
        return bool(result_path)
    except Exception as e:
        logger.error(f"Error during modeling code generation: {e}", exc_info=True)
        return False

def run_step_5_assembly(dataset_id: str) -> bool:
    """Wrapper for the final code assembly step."""
    logger.info(f"STEP 5: Assembling Final Script for dataset {dataset_id}")
    try:
        project_name = f"dataset_{dataset_id}_full_pipeline"
        output_dir = Path("final_pipelines")
        
        modeling_file = Path("generated_code") / f"modeling_dataset_{dataset_id}.py"
        if not modeling_file.exists():
            logger.error(f"Modeling file not found, cannot assemble final script: {modeling_file}")
            return False

        # The modeling file already contains the preprocessing code, so we only need it.
        stage_files = [modeling_file]

        assembler = CodeAssembler()
        assembler.assemble(project_name, stage_files, output_dir)
        logger.info(f"Final assembled script created in '{output_dir}' directory.")
        return True
    except Exception as e:
        logger.error(f"Error during code assembly: {e}", exc_info=True)
        return False

# ==============================================================================
# MAIN ORCHESTRATOR
# ==============================================================================

def run_pipeline_for_id(dataset_id: str) -> bool:
    """
    Runs the full pipeline (Steps 1-5) for a given dataset ID.
    Returns True on success, False on failure.
    """
    
    logger.info(f"\nSTARTING FULL AUTOML PIPELINE FOR DATASET {dataset_id}")
    logger.info("="*80)
    start_time = datetime.now()
    
    steps = [
        (run_step_1_profiling, "Profiling"),
        (run_step_2_guidelines, "Guidelines Generation"),
        (run_step_3_preprocessing, "Preprocessing Code Generation"),
        (run_step_4_modeling, "Modeling Code Generation"),
        (run_step_5_assembly, "Code Assembly"),
    ]
    
    for step_func, step_name in steps:
        logger.info(f"--- Running Step: {step_name} ---")
        # For profiling, we need to pass the meta_file argument
        if step_name == "Profiling":
            success = step_func(dataset_id, 'meta-data.json', 'profiling_results')
        else:
            success = step_func(dataset_id)

        if not success:
            logger.error(f"PIPELINE FAILED at Step: {step_name}")
            return False

    end_time = datetime.now()
    logger.info(f"\nPIPELINE COMPLETED SUCCESSFULLY FOR DATASET {dataset_id}!")
    logger.info("="*80)
    logger.info(f"Total time: {end_time - start_time}")
    final_script_path = f"final_pipelines/dataset_{dataset_id}_full_pipeline.py"
    logger.info(f"Final executable script located at: {final_script_path}")
    logger.info(f"To run the complete model, execute:\n  python {final_script_path}")
    
    return True

def main():
    """Main execution function with argument parsing for different modes."""
    parser = argparse.ArgumentParser(
        description="Full AutoML Pipeline Orchestrator.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--id", help="Run pipeline for an existing dataset ID.")
    group.add_argument("--path", help="Run pipeline for a new dataset at the given folder path.")

    args = parser.parse_args()

    # Check for required files
    if any(not Path(f).exists() for f in ["meta-data.json", ".env"]):
        logger.critical("Missing required files. Ensure 'meta-data.json' and '.env' exist.")
        sys.exit(1)

    try:
        pipeline_success = False
        if args.id:
            pipeline_success = run_pipeline_for_id(args.id)
        
        elif args.path:
            dataset_path = Path(args.path)
            if not dataset_path.is_dir():
                logger.critical(f"Error: Provided path '{args.path}' is not a valid directory.")
                sys.exit(1)
            
            # Step 0: Generate metadata for the new path by calling the imported function
            logger.info("STEP 0: Generating Metadata for new dataset")
            new_dataset_id = generate_metadata_for_path(dataset_path, 'meta-data.json')
            
            if new_dataset_id:
                # Run the rest of the pipeline with the new ID
                pipeline_success = run_pipeline_for_id(new_dataset_id)
            else:
                logger.critical("Could not generate metadata for the new path. Pipeline halted.")
        
        # Determine final exit code based on pipeline result
        sys.exit(0 if pipeline_success else 1)

    except KeyboardInterrupt:
        logger.info("\nPipeline interrupted by user.")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"A critical error occurred in the pipeline: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
