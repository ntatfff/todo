import papermill as pm

# Định nghĩa danh sách notebook kèm theo tham số riêng cho từng file
tasks = [
    {
        "input": "GradientBoostedTreesModel.Tensorflow.ipynb",
        "output": "output/GradientBoostedTreesModel.100.gldm.Tensorflow.ipynb",
        "params": {"datasetPath": "dataset/gldm/kernel5-radius5/100.dataset.csv"}
    },
    {
        "input": "GradientBoostedTreesModel.Tensorflow.ipynb",
        "output": "output/GradientBoostedTreesModel.200.gldm.Tensorflow.ipynb",
        "params": {"datasetPath": "dataset/gldm/kernel5-radius5/200.dataset.csv"}
    },
    {
        "input": "GradientBoostedTreesModel.Tensorflow.ipynb",
        "output": "output/GradientBoostedTreesModel.300.gldm.Tensorflow.ipynb",
        "params": {"datasetPath": "dataset/gldm/kernel5-radius5/300.dataset.csv"}
    },
    {
        "input": "GradientBoostedTreesModel.Tensorflow.ipynb",
        "output": "output/GradientBoostedTreesModel.400.gldm.Tensorflow.ipynb",
        "params": {"datasetPath": "dataset/gldm/kernel5-radius5/400.dataset.csv"}
    },
    {
        "input": "GradientBoostedTreesModel.Tensorflow.ipynb",
        "output": "output/GradientBoostedTreesModel.500.gldm.Tensorflow.ipynb",
        "params": {"datasetPath": "dataset/gldm/kernel5-radius5/500.dataset.csv"}
    },
    {
        "input": "RandomForest.Tensorflow.ipynb",
        "output": "output/RandomForest.100.gldm.Tensorflow.ipynb",
        "params": {"datasetPath": "dataset/gldm/kernel5-radius5/100.dataset.csv"}
    },
    {
        "input": "RandomForest.Tensorflow.ipynb",
        "output": "output/RandomForest.200.gldm.Tensorflow.ipynb",
        "params": {"datasetPath": "dataset/gldm/kernel5-radius5/200.dataset.csv"}
    },
    {
        "input": "RandomForest.Tensorflow.ipynb",
        "output": "output/RandomForest.300.gldm.Tensorflow.ipynb",
        "params": {"datasetPath": "dataset/gldm/kernel5-radius5/300.dataset.csv"}
    },
    {
        "input": "RandomForest.Tensorflow.ipynb",
        "output": "output/RandomForest.400.gldm.Tensorflow.ipynb",
        "params": {"datasetPath": "dataset/gldm/kernel5-radius5/400.dataset.csv"}
    },
    {
        "input": "RandomForest.Tensorflow.ipynb",
        "output": "output/RandomForest.500.gldm.Tensorflow.ipynb",
        "params": {"datasetPath": "dataset/gldm/kernel5-radius5/500.dataset.csv"}
    }
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