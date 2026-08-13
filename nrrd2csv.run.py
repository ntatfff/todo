import papermill as pm

# Định nghĩa danh sách notebook kèm theo tham số riêng cho từng file
tasks = [
    {
        "input": "nrrd2csv.ipynb",
        "output": "nrrd2csv.firstorder.kernel3.ipynb",
        "params": {"rootPath": "dataset/firstorder/kernel3/"}
    },
    {
        "input": "nrrd2csv.ipynb",
        "output": "nrrd2csv.glcm.kernel3.ipynb",
        "params": {"rootPath": "dataset/glcm/kernel3/"}
    },
    {
        "input": "nrrd2csv.ipynb",
        "output": "nrrd2csv.gldm.kernel3.ipynb",
        "params": {"rootPath": "dataset/gldm/kernel3/"}
    },
    {
        "input": "nrrd2csv.ipynb",
        "output": "nrrd2csv.glrlm.kernel3.ipynb",
        "params": {"rootPath": "dataset/glrlm/kernel3/"}
    },
    {
        "input": "nrrd2csv.ipynb",
        "output": "nrrd2csv.glszm.kernel3.ipynb",
        "params": {"rootPath": "dataset/glszm/kernel3/"}
    },
    {
        "input": "nrrd2csv.ipynb",
        "output": "nrrd2csv.ngtdm.kernel3.ipynb",
        "params": {"rootPath": "dataset/ngtdm/kernel3/"}
    },
    {
        "input": "nrrd2csv.ipynb",
        "output": "nrrd2csv.firstorder.kernel5.ipynb",
        "params": {"rootPath": "dataset/firstorder/kernel5/"}
    },
    {
        "input": "nrrd2csv.ipynb",
        "output": "nrrd2csv.glcm.kernel5.ipynb",
        "params": {"rootPath": "dataset/glcm/kernel5/"}
    },
    {
        "input": "nrrd2csv.ipynb",
        "output": "nrrd2csv.gldm.kernel5.ipynb",
        "params": {"rootPath": "dataset/gldm/kernel5/"}
    },
    {
        "input": "nrrd2csv.ipynb",
        "output": "nrrd2csv.glrlm.kernel5.ipynb",
        "params": {"rootPath": "dataset/glrlm/kernel5/"}
    },
    {
        "input": "nrrd2csv.ipynb",
        "output": "nrrd2csv.glszm.kernel5.ipynb",
        "params": {"rootPath": "dataset/glszm/kernel5/"}
    },
    {
        "input": "nrrd2csv.ipynb",
        "output": "nrrd2csv.ngtdm.kernel5.ipynb",
        "params": {"rootPath": "dataset/ngtdm/kernel5/"}
    },
]

# Chạy tuần tự các notebook với tham số tương ứng
for task in tasks:
    print(f"Đang chạy {task['input']} với rootPath = {task['params']['rootPath']}...")
    
    pm.execute_notebook(
        input_path=task["input"],
        output_path=task["output"],
        parameters=task["params"]
    )

    print(f"Hoàn thành {task['input']}\n")