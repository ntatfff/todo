import papermill as pm

# Định nghĩa danh sách notebook kèm theo tham số riêng cho từng file
tasks = [
    {
        "input": "Train.XGBoost.1200p.thresholdX.ipynb",
        "output": "./output/thresholdX/kernel5/XGBoost.1200p.threshold086.ipynb",
        "params": {
            "datasetPath": "dataset/1200p/glcm/kernel5/",
            "threshold": 0.86
        }
    },
    {
        "input": "Train.XGBoost.1200p.thresholdX.ipynb",
        "output": "./output/thresholdX/kernel5/XGBoost.1200p.threshold087.ipynb",
        "params": {
            "datasetPath": "dataset/1200p/glcm/kernel5/",
            "threshold": 0.87
        }
    },
    {
        "input": "Train.XGBoost.1200p.thresholdX.ipynb",
        "output": "./output/thresholdX/kernel5/XGBoost.1200p.threshold088.ipynb",
        "params": {
            "datasetPath": "dataset/1200p/glcm/kernel5/",
            "threshold": 0.88
        }
    },
    {
        "input": "Train.XGBoost.1200p.thresholdX.ipynb",
        "output": "./output/thresholdX/kernel5/XGBoost.1200p.threshold089.ipynb",
        "params": {
            "datasetPath": "dataset/1200p/glcm/kernel5/",
            "threshold": 0.89
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
        "output": "./output/thresholdX/kernel5/XGBoost.1200p.threshold091.ipynb",
        "params": {
            "datasetPath": "dataset/1200p/glcm/kernel5/",
            "threshold": 0.91
        }
    },
    {
        "input": "Train.XGBoost.1200p.thresholdX.ipynb",
        "output": "./output/thresholdX/kernel5/XGBoost.1200p.threshold092.ipynb",
        "params": {
            "datasetPath": "dataset/1200p/glcm/kernel5/",
            "threshold": 0.92
        }
    },
        {
            "input": "Train.XGBoost.1200p.thresholdX.ipynb",
            "output": "./output/thresholdX/kernel5/XGBoost.1200p.threshold093.ipynb",
            "params": {
                "datasetPath": "dataset/1200p/glcm/kernel5/",
                "threshold": 0.93
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