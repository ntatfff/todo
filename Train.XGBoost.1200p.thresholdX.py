import papermill as pm

# Định nghĩa danh sách notebook kèm theo tham số riêng cho từng file
tasks = [
    {
        "input": "Train.XGBoost.1200p.thresholdX.ipynb",
        "output": "./output/thresholdX/kernel5/XGBoost.1200p.threshold005.ipynb",
        "params": {
            "datasetPath": "dataset/1200p/glcm/kernel5/",
            "threshold": 0.05
        }
    },
    {
        "input": "Train.XGBoost.1200p.thresholdX.ipynb",
        "output": "./output/thresholdX/kernel5/XGBoost.1200p.threshold010.ipynb",
        "params": {
            "datasetPath": "dataset/1200p/glcm/kernel5/",
            "threshold": 0.10
        }
    },
    {
        "input": "Train.XGBoost.1200p.thresholdX.ipynb",
        "output": "./output/thresholdX/kernel5/XGBoost.1200p.threshold015.ipynb",
        "params": {
            "datasetPath": "dataset/1200p/glcm/kernel5/",
            "threshold": 0.15
        }
    },
    {
        "input": "Train.XGBoost.1200p.thresholdX.ipynb",
        "output": "./output/thresholdX/kernel5/XGBoost.1200p.threshold020.ipynb",
        "params": {
            "datasetPath": "dataset/1200p/glcm/kernel5/",
            "threshold": 0.20
        }
    },
    {
        "input": "Train.XGBoost.1200p.thresholdX.ipynb",
        "output": "./output/thresholdX/kernel5/XGBoost.1200p.threshold025.ipynb",
        "params": {
            "datasetPath": "dataset/1200p/glcm/kernel5/",
            "threshold": 0.25
        }
    },
    {
        "input": "Train.XGBoost.1200p.thresholdX.ipynb",
        "output": "./output/thresholdX/kernel5/XGBoost.1200p.threshold030.ipynb",
        "params": {
            "datasetPath": "dataset/1200p/glcm/kernel5/",
            "threshold": 0.30
        }
    },
    {
        "input": "Train.XGBoost.1200p.thresholdX.ipynb",
        "output": "./output/thresholdX/kernel5/XGBoost.1200p.threshold035.ipynb",
        "params": {
            "datasetPath": "dataset/1200p/glcm/kernel5/",
            "threshold": 0.35
        }
    },
    {
        "input": "Train.XGBoost.1200p.thresholdX.ipynb",
        "output": "./output/thresholdX/kernel5/XGBoost.1200p.threshold040.ipynb",
        "params": {
            "datasetPath": "dataset/1200p/glcm/kernel5/",
            "threshold": 0.40
        }
    },
    {
        "input": "Train.XGBoost.1200p.thresholdX.ipynb",
        "output": "./output/thresholdX/kernel5/XGBoost.1200p.threshold045.ipynb",
        "params": {
            "datasetPath": "dataset/1200p/glcm/kernel5/",
            "threshold": 0.45
        }
    },
    {
        "input": "Train.XGBoost.1200p.thresholdX.ipynb",
        "output": "./output/thresholdX/kernel5/XGBoost.1200p.threshold080.ipynb",
        "params": {
            "datasetPath": "dataset/1200p/glcm/kernel5/",
            "threshold": 0.80
        }
    },
    {
        "input": "Train.XGBoost.1200p.thresholdX.ipynb",
        "output": "./output/thresholdX/kernel5/XGBoost.1200p.threshold085.ipynb",
        "params": {
            "datasetPath": "dataset/1200p/glcm/kernel5/",
            "threshold": 0.85
        }
    },
    {
        "input": "Train.XGBoost.1200p.thresholdX.ipynb",
        "output": "./output/thresholdX/kernel5/XGBoost.1200p.threshold090.ipynb",
        "params": {
            "datasetPath": "dataset/1200p/glcm/kernel5/",
            "threshold": 0.90
        }
    },
    {
        "input": "Train.XGBoost.1200p.thresholdX.ipynb",
        "output": "./output/thresholdX/kernel5/XGBoost.1200p.threshold095.ipynb",
        "params": {
            "datasetPath": "dataset/1200p/glcm/kernel5/",
            "threshold": 0.95
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