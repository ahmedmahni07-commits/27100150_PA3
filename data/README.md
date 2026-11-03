# data/

Committed: small selection/ID files written by the scripts (`task1/`, `task2/`) and `first_plot_questions.yaml`
(copied verbatim from github.com/emergent-misalignment/emergent-misalignment, `evaluation/first_plot_questions.yaml`).

Not committed (see `.gitignore`):
* `ATML-PA3-Data/` – the course rollout export (Google Drive). Expected layout:
  `ATML-PA3-Data/Qwen2.5-3B-Instruct/{generations,probability_metrics}.parquet` and
  `ATML-PA3-Data/Qwen2.5-VL-3B-Instruct/{generations,probability_metrics,attention_metrics}.parquet`
  (rename the folders in `configs/task3.yaml:models` if the export uses different names).
* `mathvista_testmini.json` – `https://raw.githubusercontent.com/lupantech/MathVista/main/data/testmini.json`
* `task2_raw/`, `task2_images/` – COCO 2014 / VQAv2 annotations and the selected images (downloaded by `task2_vlm.data`).
