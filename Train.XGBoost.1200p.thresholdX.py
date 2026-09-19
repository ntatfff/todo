import papermill as pm

# Định nghĩa danh sách notebook kèm theo tham số riêng cho từng file
tasks = [
    {
        "input": "Train.XGBoost.1200p.thresholdX.ipynb",
        "output": "./output/thresholdX/kernel5/XGBoost.1200p.threshold050.ipynb",
        "params": {
            "datasetPath": "dataset/1200p/glcm/kernel5/",
            "threshold": 0.50
        }
    },
    {
        "input": "Train.XGBoost.1200p.thresholdX.ipynb",
        "output": "./output/thresholdX/kernel5/XGBoost.1200p.threshold055.ipynb",
        "params": {
            "datasetPath": "dataset/1200p/glcm/kernel5/",
            "threshold": 0.55
        }
    },
    {
        "input": "Train.XGBoost.1200p.thresholdX.ipynb",
        "output": "./output/thresholdX/kernel5/XGBoost.1200p.threshold060.ipynb",
        "params": {
            "datasetPath": "dataset/1200p/glcm/kernel5/",
            "threshold": 0.60
        }
    },
    {
        "input": "Train.XGBoost.1200p.thresholdX.ipynb",
        "output": "./output/thresholdX/kernel5/XGBoost.1200p.threshold065.ipynb",
        "params": {
            "datasetPath": "dataset/1200p/glcm/kernel5/",
            "threshold": 0.65
        }
    },
    {
        "input": "Train.XGBoost.1200p.thresholdX.ipynb",
        "output": "./output/thresholdX/kernel5/XGBoost.1200p.threshold070.ipynb",
        "params": {
            "datasetPath": "dataset/1200p/glcm/kernel5/",
            "threshold": 0.70
        }
    },
    {
        "input": "Train.XGBoost.1200p.thresholdX.ipynb",
        "output": "./output/thresholdX/kernel5/XGBoost.1200p.threshold075.ipynb",
        "params": {
            "datasetPath": "dataset/1200p/glcm/kernel5/",
            "threshold": 0.75
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