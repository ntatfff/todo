import papermill as pm

# Định nghĩa danh sách notebook kèm theo tham số riêng cho từng file
tasks = [
    {
        "input": "Train.XGBoost.ipynb",
        "output": "./output/mask5/10p/kernel5/XGBoost.all.ipynb",
        "params": {
            "datasetPath": "dataset/mask5/10p/all/kernel5/",
        }
    },
    {
        "input": "Train.XGBoost.ipynb",
        "output": "./output/mask5/10p/kernel5/XGBoost.firstorder.ipynb",
        "params": {
            "datasetPath": "dataset/mask5/10p/firstorder/kernel5/",
        }
    },
    {
        "input": "Train.XGBoost.ipynb",
        "output": "./output/mask5/10p/kernel5/XGBoost.glcm.ipynb",
        "params": {
            "datasetPath": "dataset/mask5/10p/glcm/kernel5/",
        }
    },
    {
        "input": "Train.XGBoost.ipynb",
        "output": "./output/mask5/10p/kernel5/XGBoost.gldm.ipynb",
        "params": {
            "datasetPath": "dataset/mask5/10p/gldm/kernel5/",
        }
    },
    {
        "input": "Train.XGBoost.ipynb",
        "output": "./output/mask5/10p/kernel5/XGBoost.glrlm.ipynb",
        "params": {
            "datasetPath": "dataset/mask5/10p/glrlm/kernel5/",
        }
    },
    {
        "input": "Train.XGBoost.ipynb",
        "output": "./output/mask5/10p/kernel5/XGBoost.glszm.ipynb",
        "params": {
            "datasetPath": "dataset/mask5/10p/glszm/kernel5/",
        }
    },
    {
        "input": "Train.XGBoost.ipynb",
        "output": "./output/mask5/10p/kernel5/XGBoost.ngtdm.ipynb",
        "params": {
            "datasetPath": "dataset/mask5/10p/ngtdm/kernel5/",
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