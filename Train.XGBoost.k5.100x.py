import papermill as pm

# Định nghĩa danh sách notebook kèm theo tham số riêng cho từng file
tasks = [
    {
        "input": "Train.XGBoost.240t.ipynb",
        "output": "./output/240t/kernel5/XGBoost.glcm.100p.ipynb",
        "params": {
            "datasetPath": "dataset/100p/glcm/kernel5/",
        }
    },
    {
        "input": "Train.XGBoost.240t.ipynb",
        "output": "./output/240t/kernel5/XGBoost.glcm.200p.ipynb",
        "params": {
            "datasetPath": "dataset/200p/glcm/kernel5/",
        }
    },
    {
        "input": "Train.XGBoost.240t.ipynb",
        "output": "./output/240t/kernel5/XGBoost.glcm.300p.ipynb",
        "params": {
            "datasetPath": "dataset/300p/glcm/kernel5/",
        }
    },
    {
        "input": "Train.XGBoost.240t.ipynb",
        "output": "./output/240t/kernel5/XGBoost.glcm.400p.ipynb",
        "params": {
            "datasetPath": "dataset/400p/glcm/kernel5/",
        }
    },
    {
        "input": "Train.XGBoost.240t.ipynb",
        "output": "./output/240t/kernel5/XGBoost.glcm.500p.ipynb",
        "params": {
            "datasetPath": "dataset/500p/glcm/kernel5/",
        }
    },
    {
        "input": "Train.XGBoost.240t.ipynb",
        "output": "./output/240t/kernel5/XGBoost.glcm.600p.ipynb",
        "params": {
            "datasetPath": "dataset/600p/glcm/kernel5/",
        }
    },
    {
        "input": "Train.XGBoost.240t.ipynb",
        "output": "./output/240t/kernel5/XGBoost.glcm.700p.ipynb",
        "params": {
            "datasetPath": "dataset/700p/glcm/kernel5/",
        }
    },
    {
        "input": "Train.XGBoost.240t.ipynb",
        "output": "./output/240t/kernel5/XGBoost.glcm.800p.ipynb",
        "params": {
            "datasetPath": "dataset/800p/glcm/kernel5/",
        }
    },
    {
        "input": "Train.XGBoost.240t.ipynb",
        "output": "./output/240t/kernel5/XGBoost.glcm.900p.ipynb",
        "params": {
            "datasetPath": "dataset/900p/glcm/kernel5/",
        }
    },
    {
        "input": "Train.XGBoost.240t.ipynb",
        "output": "./output/240t/kernel5/XGBoost.glcm.1000p.ipynb",
        "params": {
            "datasetPath": "dataset/1000p/glcm/kernel5/",
        }
    },
    {
        "input": "Train.XGBoost.240t.ipynb",
        "output": "./output/240t/kernel5/XGBoost.glcm.1100p.ipynb",
        "params": {
            "datasetPath": "dataset/1100p/glcm/kernel5/",
        }
    },
    {
        "input": "Train.XGBoost.240t.ipynb",
        "output": "./output/240t/kernel5/XGBoost.glcm.1200p.ipynb",
        "params": {
            "datasetPath": "dataset/1200p/glcm/kernel5/",
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