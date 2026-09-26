import papermill as pm

# Định nghĩa danh sách notebook kèm theo tham số riêng cho từng file
tasks = [
    {
        "input": "XGBoost.glcm.CV5.ipynb",
        "output": "./output/CV5/XGBoost.100p.ipynb",
        "params": {
            "datasetPath": "dataset/100p/glcm/kernel5/"
        }
    },
    {
        "input": "XGBoost.glcm.CV5.ipynb",
        "output": "./output/CV5/XGBoost.200p.ipynb",
        "params": {
            "datasetPath": "dataset/200p/glcm/kernel5/"
        }
    },
    {
        "input": "XGBoost.glcm.CV5.ipynb",
        "output": "./output/CV5/XGBoost.300p.ipynb",
        "params": {
            "datasetPath": "dataset/300p/glcm/kernel5/"
        }
    },
    {
        "input": "XGBoost.glcm.CV5.ipynb",
        "output": "./output/CV5/XGBoost.400p.ipynb",
        "params": {
            "datasetPath": "dataset/400p/glcm/kernel5/"
        }
    },
    {
        "input": "XGBoost.glcm.CV5.ipynb",
        "output": "./output/CV5/XGBoost.500p.ipynb",
        "params": {
            "datasetPath": "dataset/500p/glcm/kernel5/"
        }
    },
    {
        "input": "XGBoost.glcm.CV5.ipynb",
        "output": "./output/CV5/XGBoost.600p.ipynb",
        "params": {
            "datasetPath": "dataset/600p/glcm/kernel5/"
        }
    },
    {
        "input": "XGBoost.glcm.CV5.ipynb",
        "output": "./output/CV5/XGBoost.700p.ipynb",
        "params": {
            "datasetPath": "dataset/700p/glcm/kernel5/"
        }
    },
    {
        "input": "XGBoost.glcm.CV5.ipynb",
        "output": "./output/CV5/XGBoost.800p.ipynb",
        "params": {
            "datasetPath": "dataset/800p/glcm/kernel5/"
        }
    },
    {
        "input": "XGBoost.glcm.CV5.ipynb",
        "output": "./output/CV5/XGBoost.900p.ipynb",
        "params": {
            "datasetPath": "dataset/900p/glcm/kernel5/"
        }
    },
    {
        "input": "XGBoost.glcm.CV5.ipynb",
        "output": "./output/CV5/XGBoost.1000p.ipynb",
        "params": {
            "datasetPath": "dataset/1000p/glcm/kernel5/"
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