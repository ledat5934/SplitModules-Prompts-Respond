Steps:
1) py metadata_create.py FILE_PATH ( nên chạy ở window sau đó đổi file path trong metadata về đúng file path trên kaggle vì duyệt file trên kaggle chậm) (Đã có sẵn metadata nên k cần chạy lại)
2) py full_pipeline.py DATASET_ID ( trong metadata)
3) py code_assembler.py FILE_NAME( tên file mới được tạo ra) generated_code/preprocessing_dataset_datasetid(1, 2, ...).py generated_code/modeling_dataset_datasetid(1, 2, ...).py
   Ví dụ: code_assembler.py combined_code generated_code/preprocessing_dataset_1.py generated_code/modeling_dataset_1.py

* Đang thử prompt để chỉ execute thử trên 1 phần nhỏ dataset => nếu k gen được code thì xoá dòng dưới đây trong modeling.py và preprocessing.py:
*     13. During the quick self-test in the `if __name__ == "__main__":` block,
    read AT MOST 100 rows from every large CSV via `pd.read_csv(..., nrows=100)`
    (or `.sample(n=100)` after loading).  
    The `preprocess_data()` function itself MUST work on the full dataset
    when called by outside code.
