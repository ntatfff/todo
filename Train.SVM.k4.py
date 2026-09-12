import papermill as pm

# Định nghĩa danh sách notebook kèm theo tham số riêng cho từng file
tasks = [
    {
        "input": "Train.SVM.ipynb",
        "output": "./output/mask5/10p/kernel4/SVM.all.ipynb",
        "params": {
            "datasetPath": "dataset/mask5/10p/all/kernel4/",
        }
    },
    {
        "input": "Train.SVM.ipynb",
        "output": "./output/mask5/10p/kernel4/SVM.firstorder.ipynb",
        "params": {
            "datasetPath": "dataset/mask5/10p/firstorder/kernel4/",
        }
    },
    {
        "input": "Train.SVM.ipynb",
        "output": "./output/mask5/10p/kernel4/SVM.glcm.ipynb",
        "params": {
            "datasetPath": "dataset/mask5/10p/glcm/kernel4/",
        }
    },
    {
        "input": "Train.SVM.ipynb",
        "output": "./output/mask5/10p/kernel4/SVM.gldm.ipynb",
        "params": {
            "datasetPath": "dataset/mask5/10p/gldm/kernel4/",
        }
    },
    {
        "input": "Train.SVM.ipynb",
        "output": "./output/mask5/10p/kernel4/SVM.glrlm.ipynb",
        "params": {
            "datasetPath": "dataset/mask5/10p/glrlm/kernel4/",
        }
    },
    {
        "input": "Train.SVM.ipynb",
        "output": "./output/mask5/10p/kernel4/SVM.glszm.ipynb",
        "params": {
            "datasetPath": "dataset/mask5/10p/glszm/kernel4/",
        }
    },
    {
        "input": "Train.SVM.ipynb",
        "output": "./output/mask5/10p/kernel4/SVM.ngtdm.ipynb",
        "params": {
            "datasetPath": "dataset/mask5/10p/ngtdm/kernel4/",
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