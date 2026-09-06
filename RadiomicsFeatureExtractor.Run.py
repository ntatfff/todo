import papermill as pm

# Định nghĩa danh sách notebook kèm theo tham số riêng cho từng file
tasks = [
    {
        "input": "RadiomicsFeatureExtractor.tumor.ipynb",
        "output": "RadiomicsFeatureExtractor.kernel3.glcm.tumor.ipynb",
        "params": {"kernel": 3, "className": "glcm"}
    },
    {
        "input": "RadiomicsFeatureExtractor.non_tumor.ipynb",
        "output": "RadiomicsFeatureExtractor.kernel3.glcm.non_tumor.ipynb",
        "params": {"kernel": 3, "className": "glcm"}
    },
]

# Chạy tuần tự các notebook với tham số tương ứng
for task in tasks:
    print(f"Đang chạy {task['input']} với kernel = {task['params']['kernel']} và className = {task['params']['className']}")
    
    pm.execute_notebook(
        input_path=task["input"],
        output_path=task["output"],
        parameters=task["params"]
    )

    print(f"Hoàn thành {task['input']}\n")