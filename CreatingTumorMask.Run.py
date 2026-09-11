import papermill as pm

# Định nghĩa danh sách notebook kèm theo tham số riêng cho từng file
tasks = [
    {
        "input": "CreatingNonTumorMaskKernel4.ipynb",
        "output": "CreatingNonTumorMaskKernel4.ipynb"
    },
    {
        "input": "CreatingTumorMaskKernel4.ipynb",
        "output": "CreatingTumorMaskKernel4.ipynb"
    },
    {
        "input": "CreatingNonTumorMaskKernel5.ipynb",
        "output": "CreatingNonTumorMaskKernel5.ipynb"
    },
    {
        "input": "CreatingTumorMaskKernel5.ipynb",
        "output": "CreatingTumorMaskKernel5.ipynb"
    },
]

# Chạy tuần tự các notebook với tham số tương ứng
for task in tasks:
    print(f"Đang chạy {task['input']}")
    
    pm.execute_notebook(
        input_path=task["input"],
        output_path=task["output"]
    )

    print(f"Hoàn thành {task['input']}\n")