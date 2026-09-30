import papermill as pm

# Định nghĩa danh sách notebook kèm theo tham số riêng cho từng file
tasks = [
    {
        "input": "RadiomicsFeatureExtractor.ipynb",
        "output": "./output/RadiomicsFeatureExtractor/kernel5/glcm/RadiomicsFeatureExtractor.non_tumor.ipynb",
        "params": {
            "kernel": 5,
            "className": "firstorder",
            "typeOfVoxel": "non_tumor"
        }
    },
    {
        "input": "RadiomicsFeatureExtractor.ipynb",
        "output": "./output/RadiomicsFeatureExtractor/kernel5/glcm/RadiomicsFeatureExtractor.tumor.ipynb",
        "params": {
            "kernel": 5,
            "className": "firstorder",
            "typeOfVoxel": "tumor"
        }
    },
    {
        "input": "RadiomicsFeatureExtractor.ipynb",
        "output": "./output/RadiomicsFeatureExtractor/kernel5/glcm/RadiomicsFeatureExtractor.non_tumor.ipynb",
        "params": {
            "kernel": 5,
            "className": "gldm",
            "typeOfVoxel": "non_tumor"
        }
    },
    {
        "input": "RadiomicsFeatureExtractor.ipynb",
        "output": "./output/RadiomicsFeatureExtractor/kernel5/glcm/RadiomicsFeatureExtractor.tumor.ipynb",
        "params": {
            "kernel": 5,
            "className": "gldm",
            "typeOfVoxel": "tumor"
        }
    },
    {
        "input": "RadiomicsFeatureExtractor.ipynb",
        "output": "./output/RadiomicsFeatureExtractor/kernel5/glcm/RadiomicsFeatureExtractor.non_tumor.ipynb",
        "params": {
            "kernel": 5,
            "className": "glrlm",
            "typeOfVoxel": "non_tumor"
        }
    },
    {
        "input": "RadiomicsFeatureExtractor.ipynb",
        "output": "./output/RadiomicsFeatureExtractor/kernel5/glcm/RadiomicsFeatureExtractor.tumor.ipynb",
        "params": {
            "kernel": 5,
            "className": "glrlm",
            "typeOfVoxel": "tumor"
        }
    },
    {
        "input": "RadiomicsFeatureExtractor.ipynb",
        "output": "./output/RadiomicsFeatureExtractor/kernel5/glcm/RadiomicsFeatureExtractor.non_tumor.ipynb",
        "params": {
            "kernel": 5,
            "className": "glszm",
            "typeOfVoxel": "non_tumor"
        }
    },
    {
        "input": "RadiomicsFeatureExtractor.ipynb",
        "output": "./output/RadiomicsFeatureExtractor/kernel5/glcm/RadiomicsFeatureExtractor.tumor.ipynb",
        "params": {
            "kernel": 5,
            "className": "glszm",
            "typeOfVoxel": "tumor"
        }
    },
    {
        "input": "RadiomicsFeatureExtractor.ipynb",
        "output": "./output/RadiomicsFeatureExtractor/kernel5/glcm/RadiomicsFeatureExtractor.non_tumor.ipynb",
        "params": {
            "kernel": 5,
            "className": "ngtdm",
            "typeOfVoxel": "non_tumor"
        }
    },
    {
        "input": "RadiomicsFeatureExtractor.ipynb",
        "output": "./output/RadiomicsFeatureExtractor/kernel5/glcm/RadiomicsFeatureExtractor.tumor.ipynb",
        "params": {
            "kernel": 5,
            "className": "ngtdm",
            "typeOfVoxel": "tumor"
        }
    },
]

# Chạy tuần tự các notebook với tham số tương ứng
for task in tasks:
    print(f"Đang chạy {task['input']} với kernel = {task['params']['kernel']} và className = {task['params']['className']}")
    
    pm.execute_notebook(
        input_path=task["input"],
        output_path=task["output"],
        parameters=task["params"]
    )

    print(f"Hoàn thành {task['input']}\n")