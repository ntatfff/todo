import papermill as pm

# Định nghĩa danh sách notebook kèm theo tham số riêng cho từng file
tasks = [
    {
        "input": "Train.XGBoost.ipynb",
        "output": "./output/10p/kernel3/XGBoost.all.ipynb",
        "params": {
            "datasetPath": "dataset/10p/all/kernel3/",
        }
    },
    {
        "input": "Train.XGBoost.ipynb",
        "output": "./output/10p/kernel3/XGBoost.firstorder.ipynb",
        "params": {
            "datasetPath": "dataset/10p/firstorder/kernel3/",
        }
    },
    {
        "input": "Train.XGBoost.ipynb",
        "output": "./output/10p/kernel3/XGBoost.glcm.ipynb",
        "params": {
            "datasetPath": "dataset/10p/glcm/kernel3/",
        }
    },
    {
        "input": "Train.XGBoost.ipynb",
        "output": "./output/10p/kernel3/XGBoost.gldm.ipynb",
        "params": {
            "datasetPath": "dataset/10p/gldm/kernel3/",
        }
    },
    {
        "input": "Train.XGBoost.ipynb",
        "output": "./output/10p/kernel3/XGBoost.glrlm.ipynb",
        "params": {
            "datasetPath": "dataset/10p/glrlm/kernel3/",
        }
    },
    {
        "input": "Train.XGBoost.ipynb",
        "output": "./output/10p/kernel3/XGBoost.glszm.ipynb",
        "params": {
            "datasetPath": "dataset/10p/glszm/kernel3/",
        }
    },
    {
        "input": "Train.XGBoost.ipynb",
        "output": "./output/10p/kernel3/XGBoost.ngtdm.ipynb",
        "params": {
            "datasetPath": "dataset/10p/ngtdm/kernel3/",
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