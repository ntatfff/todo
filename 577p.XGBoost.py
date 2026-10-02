import papermill as pm

# Định nghĩa danh sách notebook kèm theo tham số riêng cho từng file
tasks = [
    {
        "input": "577p.XGBoost.ipynb",
        "output": "./577p.XGBoost.all.ipynb",
        "params": {
            "datasetPath": "dataset/577p/all/kernel5/"
        }
    },
    {
        "input": "577p.XGBoost.ipynb",
        "output": "./577p.XGBoost.firstorder.ipynb",
        "params": {
            "datasetPath": "dataset/577p/firstorder/kernel5/"
        }
    },
    {
        "input": "577p.XGBoost.ipynb",
        "output": "./577p.XGBoost.gldm.ipynb",
        "params": {
            "datasetPath": "dataset/577p/gldm/kernel5/"
        }
    },
    {
        "input": "577p.XGBoost.ipynb",
        "output": "./577p.XGBoost.glrlm.ipynb",
        "params": {
            "datasetPath": "dataset/577p/glrlm/kernel5/"
        }
    },
    {
        "input": "577p.XGBoost.ipynb",
        "output": "./577p.XGBoost.glszm.ipynb",
        "params": {
            "datasetPath": "dataset/577p/glszm/kernel5/"
        }
    },
    {
        "input": "577p.XGBoost.ipynb",
        "output": "./577p.XGBoost.ngtdm.ipynb",
        "params": {
            "datasetPath": "dataset/577p/ngtdm/kernel5/"
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