import papermill as pm

# Định nghĩa danh sách notebook kèm theo tham số riêng cho từng file
tasks = [
    {
        "input": "RadiomicsFeatureExtractor.m5.ipynb",
        "output": "./output/RadiomicsFeatureExtractor/kernel4/firstorder/RadiomicsFeatureExtractor.tumor.ipynb",
        "params": {
            "kernel": 4, 
            "className": "firstorder",
            "typeOfVoxel": "tumor"
        }
    },
    {
        "input": "RadiomicsFeatureExtractor.m5.ipynb",
        "output": "./output/RadiomicsFeatureExtractor/kernel4/glcm/RadiomicsFeatureExtractor.tumor.ipynb",
        "params": {
            "kernel": 4, 
            "className": "glcm",
            "typeOfVoxel": "tumor"
        }
    },
    {
        "input": "RadiomicsFeatureExtractor.m5.ipynb",
        "output": "./output/RadiomicsFeatureExtractor/kernel4/gldm/RadiomicsFeatureExtractor.tumor.ipynb",
        "params": {
            "kernel": 4, 
            "className": "gldm",
            "typeOfVoxel": "tumor"
        }
    },
    {
        "input": "RadiomicsFeatureExtractor.m5.ipynb",
        "output": "./output/RadiomicsFeatureExtractor/kernel4/glrlm/RadiomicsFeatureExtractor.tumor.ipynb",
        "params": {
            "kernel": 4, 
            "className": "glrlm",
            "typeOfVoxel": "tumor"
        }
    },
    {
        "input": "RadiomicsFeatureExtractor.m5.ipynb",
        "output": "./output/RadiomicsFeatureExtractor/kernel4/glszm/RadiomicsFeatureExtractor.tumor.ipynb",
        "params": {
            "kernel": 4, 
            "className": "glszm",
            "typeOfVoxel": "tumor"
        }
    },
    {
        "input": "RadiomicsFeatureExtractor.m5.ipynb",
        "output": "./output/RadiomicsFeatureExtractor/kernel4/ngtdm/RadiomicsFeatureExtractor.tumor.ipynb",
        "params": {
            "kernel": 4, 
            "className": "ngtdm",
            "typeOfVoxel": "tumor"
        }
    },
    {
        "input": "RadiomicsFeatureExtractor.m5.ipynb",
        "output": "./output/RadiomicsFeatureExtractor/kernel4/firstorder/RadiomicsFeatureExtractor.non_tumor.ipynb",
        "params": {
            "kernel": 4, 
            "className": "firstorder",
            "typeOfVoxel": "non_tumor"
        }
    },
    {
        "input": "RadiomicsFeatureExtractor.m5.ipynb",
        "output": "./output/RadiomicsFeatureExtractor/kernel4/glcm/RadiomicsFeatureExtractor.non_tumor.ipynb",
        "params": {
            "kernel": 4, 
            "className": "glcm",
            "typeOfVoxel": "non_tumor"
        }
    },
    {
        "input": "RadiomicsFeatureExtractor.m5.ipynb",
        "output": "./output/RadiomicsFeatureExtractor/kernel4/gldm/RadiomicsFeatureExtractor.non_tumor.ipynb",
        "params": {
            "kernel": 4, 
            "className": "gldm",
            "typeOfVoxel": "non_tumor"
        }
    },
    {
        "input": "RadiomicsFeatureExtractor.m5.ipynb",
        "output": "./output/RadiomicsFeatureExtractor/kernel4/glrlm/RadiomicsFeatureExtractor.non_tumor.ipynb",
        "params": {
            "kernel": 4, 
            "className": "glrlm",
            "typeOfVoxel": "non_tumor"
        }
    },
    {
        "input": "RadiomicsFeatureExtractor.m5.ipynb",
        "output": "./output/RadiomicsFeatureExtractor/kernel4/glszm/RadiomicsFeatureExtractor.non_tumor.ipynb",
        "params": {
            "kernel": 4, 
            "className": "glszm",
            "typeOfVoxel": "non_tumor"
        }
    },
    {
        "input": "RadiomicsFeatureExtractor.m5.ipynb",
        "output": "./output/RadiomicsFeatureExtractor/kernel4/ngtdm/RadiomicsFeatureExtractor.non_tumor.ipynb",
        "params": {
            "kernel": 4, 
            "className": "ngtdm",
            "typeOfVoxel": "non_tumor"
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