import papermill as pm

# Định nghĩa danh sách notebook kèm theo tham số riêng cho từng file
tasks = [
    {
        "input": "577p.nrrd2csv.ipynb",
        "output": "./577p.nrrd2csv.all.ipynb",
        "params": {
            "batchSize": 577,
            "className": "all",
        }
    },
    {
        "input": "577p.nrrd2csv.ipynb",
        "output": "./577p.nrrd2csv.firstorder.ipynb",
        "params": {
            "batchSize": 577,
            "className": "firstorder",
        }
    },
    {
        "input": "577p.nrrd2csv.ipynb",
        "output": "./577p.nrrd2csv.glcm.ipynb",
        "params": {
            "batchSize": 577,
            "className": "glcm",
        }
    },
    {
        "input": "577p.nrrd2csv.ipynb",
        "output": "./577p.nrrd2csv.gldm.ipynb",
        "params": {
            "batchSize": 577,
            "className": "gldm",
        }
    },
    {
        "input": "577p.nrrd2csv.ipynb",
        "output": "./577p.nrrd2csv.glrlm.ipynb",
        "params": {
            "batchSize": 577,
            "className": "glrlm",
        }
    },
    {
        "input": "577p.nrrd2csv.ipynb",
        "output": "./577p.nrrd2csv.glszm.ipynb",
        "params": {
            "batchSize": 577,
            "className": "glszm",
        }
    },
    {
        "input": "577p.nrrd2csv.ipynb",
        "output": "./577p.nrrd2csv.ngtdm.ipynb",
        "params": {
            "batchSize": 577,
            "className": "ngtdm",
        }
    },
]

# Chạy tuần tự các notebook với tham số tương ứng
for task in tasks:
    print(f"Đang chạy {task['input']} với batchSize = {task['params']['batchSize']}")

    pm.execute_notebook(
        input_path=task["input"],
        output_path=task["output"],
        parameters=task["params"]
    )

    print(f"Hoàn thành {task['input']}\n")