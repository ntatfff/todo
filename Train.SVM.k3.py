import papermill as pm

# Định nghĩa danh sách notebook kèm theo tham số riêng cho từng file
tasks = [
    {
        "input": "Train.SVM.ipynb",
        "output": "./output/mask5/10p/kernel3/SVM.all.ipynb",
        "params": {
            "datasetPath": "dataset/mask5/10p/all/kernel3/",
        }
    },
    {
        "input": "Train.SVM.ipynb",
        "output": "./output/mask5/10p/kernel3/SVM.firstorder.ipynb",
        "params": {
            "datasetPath": "dataset/mask5/10p/firstorder/kernel3/",
        }
    },
    {
        "input": "Train.SVM.ipynb",
        "output": "./output/mask5/10p/kernel3/SVM.glcm.ipynb",
        "params": {
            "datasetPath": "dataset/mask5/10p/glcm/kernel3/",
        }
    },
    {
        "input": "Train.SVM.ipynb",
        "output": "./output/mask5/10p/kernel3/SVM.gldm.ipynb",
        "params": {
            "datasetPath": "dataset/mask5/10p/gldm/kernel3/",
        }
    },
    {
        "input": "Train.SVM.ipynb",
        "output": "./output/mask5/10p/kernel3/SVM.glrlm.ipynb",
        "params": {
            "datasetPath": "dataset/mask5/10p/glrlm/kernel3/",
        }
    },
    {
        "input": "Train.SVM.ipynb",
        "output": "./output/mask5/10p/kernel3/SVM.glszm.ipynb",
        "params": {
            "datasetPath": "dataset/mask5/10p/glszm/kernel3/",
        }
    },
    {
        "input": "Train.SVM.ipynb",
        "output": "./output/mask5/10p/kernel3/SVM.ngtdm.ipynb",
        "params": {
            "datasetPath": "dataset/mask5/10p/ngtdm/kernel3/",
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