import papermill as pm

# Định nghĩa danh sách notebook kèm theo tham số riêng cho từng file
tasks = [
    {
        "input": "RadiomicsFeatureExtractor.non_tumor.ipynb",
        "output": "RadiomicsFeatureExtractor.kernel3.firstorder.non_tumor.ipynb",
        "params": {"kernel": 3, "className": "firstorder"}
    },
    {
        "input": "RadiomicsFeatureExtractor.tumor.ipynb",
        "output": "RadiomicsFeatureExtractor.kernel3.firstorder.tumor.ipynb",
        "params": {"kernel": 3, "className": "firstorder"}
    },
    {
        "input": "RadiomicsFeatureExtractor.non_tumor.ipynb",
        "output": "RadiomicsFeatureExtractor.kernel5.firstorder.non_tumor.ipynb",
        "params": {"kernel": 5, "className": "firstorder"}
    },
    {
        "input": "RadiomicsFeatureExtractor.tumor.ipynb",
        "output": "RadiomicsFeatureExtractor.kernel5.firstorder.tumor.ipynb",
        "params": {"kernel": 5, "className": "firstorder"}
    },
    {
        "input": "RadiomicsFeatureExtractor.non_tumor.ipynb",
        "output": "RadiomicsFeatureExtractor.kernel3.glcm.non_tumor.ipynb",
        "params": {"kernel": 3, "className": "glcm"}
    },
    {
        "input": "RadiomicsFeatureExtractor.tumor.ipynb",
        "output": "RadiomicsFeatureExtractor.kernel3.glcm.tumor.ipynb",
        "params": {"kernel": 3, "className": "glcm"}
    },
    {
        "input": "RadiomicsFeatureExtractor.non_tumor.ipynb",
        "output": "RadiomicsFeatureExtractor.kernel5.glcm.non_tumor.ipynb",
        "params": {"kernel": 5, "className": "glcm"}
    },
    {
        "input": "RadiomicsFeatureExtractor.tumor.ipynb",
        "output": "RadiomicsFeatureExtractor.kernel5.glcm.tumor.ipynb",
        "params": {"kernel": 5, "className": "glcm"}
    },
    {
        "input": "RadiomicsFeatureExtractor.non_tumor.ipynb",
        "output": "RadiomicsFeatureExtractor.kernel3.gldm.non_tumor.ipynb",
        "params": {"kernel": 3, "className": "gldm"}
    },
    {
        "input": "RadiomicsFeatureExtractor.tumor.ipynb",
        "output": "RadiomicsFeatureExtractor.kernel3.gldm.tumor.ipynb",
        "params": {"kernel": 3, "className": "gldm"}
    },
    {
        "input": "RadiomicsFeatureExtractor.non_tumor.ipynb",
        "output": "RadiomicsFeatureExtractor.kernel5.gldm.non_tumor.ipynb",
        "params": {"kernel": 5, "className": "gldm"}
    },
    {
        "input": "RadiomicsFeatureExtractor.tumor.ipynb",
        "output": "RadiomicsFeatureExtractor.kernel5.gldm.tumor.ipynb",
        "params": {"kernel": 5, "className": "gldm"}
    },
    {
        "input": "RadiomicsFeatureExtractor.non_tumor.ipynb",
        "output": "RadiomicsFeatureExtractor.kernel3.glrlm.non_tumor.ipynb",
        "params": {"kernel": 3, "className": "glrlm"}
    },
    {
        "input": "RadiomicsFeatureExtractor.tumor.ipynb",
        "output": "RadiomicsFeatureExtractor.kernel3.glrlm.tumor.ipynb",
        "params": {"kernel": 3, "className": "glrlm"}
    },
    {
        "input": "RadiomicsFeatureExtractor.non_tumor.ipynb",
        "output": "RadiomicsFeatureExtractor.kernel5.glrlm.non_tumor.ipynb",
        "params": {"kernel": 5, "className": "glrlm"}
    },
    {
        "input": "RadiomicsFeatureExtractor.tumor.ipynb",
        "output": "RadiomicsFeatureExtractor.kernel5.glrlm.tumor.ipynb",
        "params": {"kernel": 5, "className": "glrlm"}
    },
    {
        "input": "RadiomicsFeatureExtractor.non_tumor.ipynb",
        "output": "RadiomicsFeatureExtractor.kernel3.glszm.non_tumor.ipynb",
        "params": {"kernel": 3, "className": "glszm"}
    },
    {
        "input": "RadiomicsFeatureExtractor.tumor.ipynb",
        "output": "RadiomicsFeatureExtractor.kernel3.glszm.tumor.ipynb",
        "params": {"kernel": 3, "className": "glszm"}
    },
    {
        "input": "RadiomicsFeatureExtractor.non_tumor.ipynb",
        "output": "RadiomicsFeatureExtractor.kernel5.glszm.non_tumor.ipynb",
        "params": {"kernel": 5, "className": "glszm"}
    },
    {
        "input": "RadiomicsFeatureExtractor.tumor.ipynb",
        "output": "RadiomicsFeatureExtractor.kernel5.glszm.tumor.ipynb",
        "params": {"kernel": 5, "className": "glszm"}
    },
    {
        "input": "RadiomicsFeatureExtractor.non_tumor.ipynb",
        "output": "RadiomicsFeatureExtractor.kernel3.ngtdm.non_tumor.ipynb",
        "params": {"kernel": 3, "className": "ngtdm"}
    },
    {
        "input": "RadiomicsFeatureExtractor.tumor.ipynb",
        "output": "RadiomicsFeatureExtractor.kernel3.ngtdm.tumor.ipynb",
        "params": {"kernel": 3, "className": "ngtdm"}
    },
    {
        "input": "RadiomicsFeatureExtractor.non_tumor.ipynb",
        "output": "RadiomicsFeatureExtractor.kernel5.ngtdm.non_tumor.ipynb",
        "params": {"kernel": 5, "className": "ngtdm"}
    },
    {
        "input": "RadiomicsFeatureExtractor.tumor.ipynb",
        "output": "RadiomicsFeatureExtractor.kernel5.ngtdm.tumor.ipynb",
        "params": {"kernel": 5, "className": "ngtdm"}
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