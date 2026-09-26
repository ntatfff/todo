import papermill as pm

# Định nghĩa danh sách notebook kèm theo tham số riêng cho từng file
tasks = [
    {
        "input": "nrrd2csv.glcm.test.ipynb",
        "output": "./output/nrrd2csv/nrrd2csv.glcm.ipynb",
        "params": {
            "batchSize": 100,
        }
    },
    {
        "input": "nrrd2csv.glcm.test.ipynb",
        "output": "./output/nrrd2csv/nrrd2csv.glcm.ipynb",
        "params": {
            "batchSize": 200,
        }
    },
    {
        "input": "nrrd2csv.glcm.test.ipynb",
        "output": "./output/nrrd2csv/nrrd2csv.glcm.ipynb",
        "params": {
            "batchSize": 300,
        }
    },
    {
        "input": "nrrd2csv.glcm.test.ipynb",
        "output": "./output/nrrd2csv/nrrd2csv.glcm.ipynb",
        "params": {
            "batchSize": 400,
        }
    },
    {
        "input": "nrrd2csv.glcm.test.ipynb",
        "output": "./output/nrrd2csv/nrrd2csv.glcm.ipynb",
        "params": {
            "batchSize": 500,
        }
    },
    {
        "input": "nrrd2csv.glcm.test.ipynb",
        "output": "./output/nrrd2csv/nrrd2csv.glcm.ipynb",
        "params": {
            "batchSize": 600,
        }
    },
    {
        "input": "nrrd2csv.glcm.test.ipynb",
        "output": "./output/nrrd2csv/nrrd2csv.glcm.ipynb",
        "params": {
            "batchSize": 700,
        }
    },
    {
        "input": "nrrd2csv.glcm.test.ipynb",
        "output": "./output/nrrd2csv/nrrd2csv.glcm.ipynb",
        "params": {
            "batchSize": 800,
        }
    },
    {
        "input": "nrrd2csv.glcm.test.ipynb",
        "output": "./output/nrrd2csv/nrrd2csv.glcm.ipynb",
        "params": {
            "batchSize": 900,
        }
    },
    {
        "input": "nrrd2csv.glcm.test.ipynb",
        "output": "./output/nrrd2csv/nrrd2csv.glcm.ipynb",
        "params": {
            "batchSize": 1000,
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