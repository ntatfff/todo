import papermill as pm

# Định nghĩa danh sách notebook kèm theo tham số riêng cho từng file
tasks = [
    {
        "input": "Train.XGBoost.ipynb",
        "output": "./output/kernel5/XGBoost.glcm.100p.ipynb",
        "params": {
            "datasetPath": "dataset/100p/glcm/kernel5/",
        }
    },
    {
        "input": "Train.XGBoost.ipynb",
        "output": "./output/kernel5/XGBoost.glcm.200p.ipynb",
        "params": {
            "datasetPath": "dataset/200p/glcm/kernel5/",
        }
    },
    {
        "input": "Train.XGBoost.ipynb",
        "output": "./output/kernel5/XGBoost.glcm.300p.ipynb",
        "params": {
            "datasetPath": "dataset/300p/glcm/kernel5/",
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