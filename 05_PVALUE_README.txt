Notebook: 05_Paired_Patient_Wilcoxon_Holm_577p.ipynb

Mục tiêu:
- Wilcoxon signed-rank paired theo 577 bệnh nhân
- rank-biserial effect size
- Holm correction
- xuất Bảng 4.12a và đoạn mô tả Chương 4

Preflight sandbox hiện tại:
- Balanced OOF: FOUND: /mnt/data/development_oof_predictions.csv
- None OOF: MISSING

Hiện sandbox chỉ có balanced OOF, nên chưa thể tính p-value theo 577 bệnh nhân mà không bịa số.

Trong môi trường dự án, hãy bảo đảm Notebook 01 đã tạo một trong hai dạng:
1) development_oof_predictions_none.csv + development_oof_predictions_balanced.csv
hoặc
2) class_weight_ablation_oof_predictions.csv

Sau đó chạy Notebook 05. Notebook sẽ tự audit pairing và xuất:
- 05_pairing_audit.json
- 05_per_patient_metrics_none_balanced.csv
- 05_wilcoxon_holm_all_metrics.csv
- 05_chapter4_table_4_12a.csv
- 05_chapter4_table_4_12a.md
- 05_chapter4_text_ready.txt
