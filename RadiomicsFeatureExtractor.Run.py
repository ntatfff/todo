import papermill as pm

# Định nghĩa danh sách notebook kèm theo tham số riêng cho từng file
tasks = [
    {
        "input": "RadiomicsFeatureExtractor.non_tumor.ipynb",
        "output": "RadiomicsFeatureExtractor.kernel4.firstorder.non_tumor.ipynb",
        "params": {"kernel": 4, "className": "firstorder"}
    },
    {
        "input": "RadiomicsFeatureExtractor.tumor.ipynb",
        "output": "RadiomicsFeatureExtractor.kernel4.firstorder.tumor.ipynb",
        "params": {"kernel": 4, "className": "firstorder"}
    },
    {
        "input": "RadiomicsFeatureExtractor.non_tumor.ipynb",
        "output": "RadiomicsFeatureExtractor.kernel6.firstorder.non_tumor.ipynb",
        "params": {"kernel": 6, "className": "firstorder"}
    },
    {
        "input": "RadiomicsFeatureExtractor.tumor.ipynb",
        "output": "RadiomicsFeatureExtractor.kernel6.firstorder.tumor.ipynb",
        "params": {"kernel": 6, "className": "firstorder"}
    },
    {
        "input": "RadiomicsFeatureExtractor.non_tumor.ipynb",
        "output": "RadiomicsFeatureExtractor.kernel4.glcm.non_tumor.ipynb",
        "params": {"kernel": 4, "className": "glcm"}
    },
    {
        "input": "RadiomicsFeatureExtractor.tumor.ipynb",
        "output": "RadiomicsFeatureExtractor.kernel4.glcm.tumor.ipynb",
        "params": {"kernel": 4, "className": "glcm"}
    },
    {
        "input": "RadiomicsFeatureExtractor.non_tumor.ipynb",
        "output": "RadiomicsFeatureExtractor.kernel6.glcm.non_tumor.ipynb",
        "params": {"kernel": 6, "className": "glcm"}
    },
    {
        "input": "RadiomicsFeatureExtractor.tumor.ipynb",
        "output": "RadiomicsFeatureExtractor.kernel6.glcm.tumor.ipynb",
        "params": {"kernel": 6, "className": "glcm"}
    },
    {
        "input": "RadiomicsFeatureExtractor.non_tumor.ipynb",
        "output": "RadiomicsFeatureExtractor.kernel4.gldm.non_tumor.ipynb",
        "params": {"kernel": 4, "className": "gldm"}
    },
    {
        "input": "RadiomicsFeatureExtractor.tumor.ipynb",
        "output": "RadiomicsFeatureExtractor.kernel4.gldm.tumor.ipynb",
        "params": {"kernel": 4, "className": "gldm"}
    },
    {
        "input": "RadiomicsFeatureExtractor.non_tumor.ipynb",
        "output": "RadiomicsFeatureExtractor.kernel6.gldm.non_tumor.ipynb",
        "params": {"kernel": 6, "className": "gldm"}
    },
    {
        "input": "RadiomicsFeatureExtractor.tumor.ipynb",
        "output": "RadiomicsFeatureExtractor.kernel6.gldm.tumor.ipynb",
        "params": {"kernel": 6, "className": "gldm"}
    },
    {
        "input": "RadiomicsFeatureExtractor.non_tumor.ipynb",
        "output": "RadiomicsFeatureExtractor.kernel4.glrlm.non_tumor.ipynb",
        "params": {"kernel": 4, "className": "glrlm"}
    },
    {
        "input": "RadiomicsFeatureExtractor.tumor.ipynb",
        "output": "RadiomicsFeatureExtractor.kernel4.glrlm.tumor.ipynb",
        "params": {"kernel": 4, "className": "glrlm"}
    },
    {
        "input": "RadiomicsFeatureExtractor.non_tumor.ipynb",
        "output": "RadiomicsFeatureExtractor.kernel6.glrlm.non_tumor.ipynb",
        "params": {"kernel": 6, "className": "glrlm"}
    },
    {
        "input": "RadiomicsFeatureExtractor.tumor.ipynb",
        "output": "RadiomicsFeatureExtractor.kernel6.glrlm.tumor.ipynb",
        "params": {"kernel": 6, "className": "glrlm"}
    },
    {
        "input": "RadiomicsFeatureExtractor.non_tumor.ipynb",
        "output": "RadiomicsFeatureExtractor.kernel4.glszm.non_tumor.ipynb",
        "params": {"kernel": 4, "className": "glszm"}
    },
    {
        "input": "RadiomicsFeatureExtractor.tumor.ipynb",
        "output": "RadiomicsFeatureExtractor.kernel4.glszm.tumor.ipynb",
        "params": {"kernel": 4, "className": "glszm"}
    },
    {
        "input": "RadiomicsFeatureExtractor.non_tumor.ipynb",
        "output": "RadiomicsFeatureExtractor.kernel6.glszm.non_tumor.ipynb",
        "params": {"kernel": 6, "className": "glszm"}
    },
    {
        "input": "RadiomicsFeatureExtractor.tumor.ipynb",
        "output": "RadiomicsFeatureExtractor.kernel6.glszm.tumor.ipynb",
        "params": {"kernel": 6, "className": "glszm"}
    },
    {
        "input": "RadiomicsFeatureExtractor.non_tumor.ipynb",
        "output": "RadiomicsFeatureExtractor.kernel4.ngtdm.non_tumor.ipynb",
        "params": {"kernel": 4, "className": "ngtdm"}
    },
    {
        "input": "RadiomicsFeatureExtractor.tumor.ipynb",
        "output": "RadiomicsFeatureExtractor.kernel4.ngtdm.tumor.ipynb",
        "params": {"kernel": 4, "className": "ngtdm"}
    },
    {
        "input": "RadiomicsFeatureExtractor.non_tumor.ipynb",
        "output": "RadiomicsFeatureExtractor.kernel6.ngtdm.non_tumor.ipynb",
        "params": {"kernel": 6, "className": "ngtdm"}
    },
    {
        "input": "RadiomicsFeatureExtractor.tumor.ipynb",
        "output": "RadiomicsFeatureExtractor.kernel6.ngtdm.tumor.ipynb",
        "params": {"kernel": 6, "className": "ngtdm"}
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