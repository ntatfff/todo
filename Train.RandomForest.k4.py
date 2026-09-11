import papermill as pm

# Định nghĩa danh sách notebook kèm theo tham số riêng cho từng file
tasks = [
    {
        "input": "Train.RandomForest.ipynb",
        "output": "./output/10p/kernel4/RandomForest.all.ipynb",
        "params": {
            "datasetPath": "dataset/10p/all/kernel4/",
        }
    },
    {
        "input": "Train.RandomForest.ipynb",
        "output": "./output/10p/kernel4/RandomForest.firstorder.ipynb",
        "params": {
            "datasetPath": "dataset/10p/firstorder/kernel4/",
        }
    },
    {
        "input": "Train.RandomForest.ipynb",
        "output": "./output/10p/kernel4/RandomForest.glcm.ipynb",
        "params": {
            "datasetPath": "dataset/10p/glcm/kernel4/",
        }
    },
    {
        "input": "Train.RandomForest.ipynb",
        "output": "./output/10p/kernel4/RandomForest.gldm.ipynb",
        "params": {
            "datasetPath": "dataset/10p/gldm/kernel4/",
        }
    },
    {
        "input": "Train.RandomForest.ipynb",
        "output": "./output/10p/kernel4/RandomForest.glrlm.ipynb",
        "params": {
            "datasetPath": "dataset/10p/glrlm/kernel4/",
        }
    },
    {
        "input": "Train.RandomForest.ipynb",
        "output": "./output/10p/kernel4/RandomForest.glszm.ipynb",
        "params": {
            "datasetPath": "dataset/10p/glszm/kernel4/",
        }
    },
    {
        "input": "Train.RandomForest.ipynb",
        "output": "./output/10p/kernel4/RandomForest.ngtdm.ipynb",
        "params": {
            "datasetPath": "dataset/10p/ngtdm/kernel4/",
        }
    },
]

# Chạy tuần tự các notebook với tham số tương ứng
for task in tasks:
    print(f"Đang chạy {task['input']} với tham số {task['params']}...")
    
    pm.execute_notebook(
        input_path=task["input"],
        output_path=task["output"],
        parameters=task["params"]  # Truyền tham số tại đây
    )

    print(f"Hoàn thành {task['input']}\n")