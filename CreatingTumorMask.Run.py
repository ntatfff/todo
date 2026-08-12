import papermill as pm

# Định nghĩa danh sách notebook kèm theo tham số riêng cho từng file
tasks = [
    # {
    #     "input": "CreatingNonTumorMaskKernel1.ipynb",
    #     "output": "CreatingNonTumorMaskKernel1.ipynb"
    # },
    {
        "input": "CreatingNonTumorMaskKernel2.ipynb",
        "output": "CreatingNonTumorMaskKernel2.ipynb"
    },
    {
        "input": "CreatingNonTumorMaskKernel3.ipynb",
        "output": "CreatingNonTumorMaskKernel3.ipynb"
    },
    {
        "input": "CreatingNonTumorMaskKernel4.ipynb",
        "output": "CreatingNonTumorMaskKernel4.ipynb"
    },
    {
        "input": "CreatingNonTumorMaskKernel6.ipynb",
        "output": "CreatingNonTumorMaskKernel6.ipynb"
    },
    {
        "input": "CreatingTumorMaskKernel1.ipynb",
        "output": "CreatingTumorMaskKernel1.ipynb"
    },
    {
        "input": "CreatingTumorMaskKernel2.ipynb",
        "output": "CreatingTumorMaskKernel2.ipynb"
    },
    {
        "input": "CreatingTumorMaskKernel3.ipynb",
        "output": "CreatingTumorMaskKernel3.ipynb"
    },
    {
        "input": "CreatingTumorMaskKernel4.ipynb",
        "output": "CreatingTumorMaskKernel4.ipynb"
    },
    {
        "input": "CreatingTumorMaskKernel6.ipynb",
        "output": "CreatingTumorMaskKernel6.ipynb"
    }
]

# Chạy tuần tự các notebook với tham số tương ứng
for task in tasks:
    print(f"Đang chạy {task['input']}")
    
    pm.execute_notebook(
        input_path=task["input"],
        output_path=task["output"]
    )

    print(f"Hoàn thành {task['input']}\n")