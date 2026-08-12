import papermill as pm

# Định nghĩa danh sách notebook kèm theo tham số riêng cho từng file
tasks = [
    {
        "input": "GradientBoostedTreesModel.Tensorflow.ipynb",
        "output": "output/kernel4/GradientBoostedTreesModel.1.firstorder.Tensorflow.ipynb",
        "params": {"datasetPath": "dataset/firstorder/kernel4-radius4/dataset.1p.csv"}
    },
    {
        "input": "GradientBoostedTreesModel.Tensorflow.ipynb",
        "output": "output/kernel4/GradientBoostedTreesModel.1.glcm.Tensorflow.ipynb",
        "params": {"datasetPath": "dataset/glcm/kernel4-radius4/dataset.1p.csv"}
    },
    {
        "input": "GradientBoostedTreesModel.Tensorflow.ipynb",
        "output": "output/kernel4/GradientBoostedTreesModel.1.gldm.Tensorflow.ipynb",
        "params": {"datasetPath": "dataset/gldm/kernel4-radius4/dataset.1p.csv"}
    },
    {
        "input": "GradientBoostedTreesModel.Tensorflow.ipynb",
        "output": "output/kernel4/GradientBoostedTreesModel.1.glrlm.Tensorflow.ipynb",
        "params": {"datasetPath": "dataset/glrlm/kernel4-radius4/dataset.1p.csv"}
    },
    {
        "input": "GradientBoostedTreesModel.Tensorflow.ipynb",
        "output": "output/kernel4/GradientBoostedTreesModel.1.glszm.Tensorflow.ipynb",
        "params": {"datasetPath": "dataset/glszm/kernel4-radius4/dataset.1p.csv"}
    },
    {
        "input": "GradientBoostedTreesModel.Tensorflow.ipynb",
        "output": "output/kernel4/GradientBoostedTreesModel.1.ngtdm.Tensorflow.ipynb",
        "params": {"datasetPath": "dataset/ngtdm/kernel4-radius4/dataset.1p.csv"}
    },
    {
        "input": "GradientBoostedTreesModel.Tensorflow.ipynb",
        "output": "output/kernel6/GradientBoostedTreesModel.1.firstorder.Tensorflow.ipynb",
        "params": {"datasetPath": "dataset/firstorder/kernel6-radius6/dataset.1p.csv"}
    },
    {
        "input": "GradientBoostedTreesModel.Tensorflow.ipynb",
        "output": "output/kernel6/GradientBoostedTreesModel.1.glcm.Tensorflow.ipynb",
        "params": {"datasetPath": "dataset/glcm/kernel6-radius6/dataset.1p.csv"}
    },
    {
        "input": "GradientBoostedTreesModel.Tensorflow.ipynb",
        "output": "output/kernel6/GradientBoostedTreesModel.1.gldm.Tensorflow.ipynb",
        "params": {"datasetPath": "dataset/gldm/kernel6-radius6/dataset.1p.csv"}
    },
    {
        "input": "GradientBoostedTreesModel.Tensorflow.ipynb",
        "output": "output/kernel6/GradientBoostedTreesModel.1.glrlm.Tensorflow.ipynb",
        "params": {"datasetPath": "dataset/glrlm/kernel6-radius6/dataset.1p.csv"}
    },
    {
        "input": "GradientBoostedTreesModel.Tensorflow.ipynb",
        "output": "output/kernel6/GradientBoostedTreesModel.1.glszm.Tensorflow.ipynb",
        "params": {"datasetPath": "dataset/glszm/kernel6-radius6/dataset.1p.csv"}
    },
    {
        "input": "GradientBoostedTreesModel.Tensorflow.ipynb",
        "output": "output/kernel6/GradientBoostedTreesModel.1.ngtdm.Tensorflow.ipynb",
        "params": {"datasetPath": "dataset/ngtdm/kernel6-radius6/dataset.1p.csv"}
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