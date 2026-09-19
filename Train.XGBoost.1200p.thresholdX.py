import papermill as pm

# Định nghĩa danh sách notebook kèm theo tham số riêng cho từng file
tasks = [
        {
            "input": "Train.XGBoost.1200p.thresholdX.ipynb",
            "output": "./output/thresholdX/kernel5/XGBoost.1200p.threshold094.ipynb",
            "params": {
                "datasetPath": "dataset/1200p/glcm/kernel5/",
                "threshold": 0.94
            }
        },
]

# Chạy tuần tự các notebook với tham số tương ứng
for task in tasks:
    print(f"Đang chạy {task['input']} với datasetPath = {task['params']['datasetPath']}")
    
    pm.execute_notebook(
        input_path=task["input"],
        output_path=task["output"],
        parameters=task["params"]
    )

    print(f"Hoàn thành {task['input']}\n")