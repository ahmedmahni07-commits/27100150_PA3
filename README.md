# 27100150 – ATML PA3

Code for the three PA3 tasks: (1) math fine-tuning of Qwen3-0.6B with Full FT / L2-SP / LoRA / OPLoRA,
(2) a SigLIP–Qwen VLM trained by a three-stage curriculum and probed for visual grounding, and (3) test-time
compute (repeated sampling, confidence-based selection, refinement, adaptive stopping).

Every script is a module run from the repository root (`python -m <package>.<module> ...`). GPU work is driven
by `notebooks/run_on_gpu.ipynb` (Colab / Kaggle), a thin driver that only calls these modules and keeps
`outputs/` (checkpoints) and `results/` (logs, CSV, JSON, figures) on persistent storage. Every training and
generation script resumes from its last saved state when re-run.

```
common/            seeding, JSON I/O, timing, GPU-memory helpers
configs/           task1.yaml, task2.yaml, task3.yaml  (all hyper-parameters; seed 6304)
task1_ft/          data.py, methods.py (LoRA/OPLoRA/L2-SP), train.py, evaluate.py, diagnostics.py, qualitative.py, summarize.py
task2_vlm/         data.py, vqa_norm.py, features.py, model.py, common.py, train.py, evaluate.py, summarize.py
task3_ttc/         data.py, grading.py, selection.py, offline.py, signals.py, live.py, adaptive.py, plots.py
tests/             CPU checks (tiny random models / synthetic rollouts; no downloads)
data/              committed selection IDs; large raw data is git-ignored (see data/README.md)
results/           machine-readable results (committed)
```

Setup: `pip install -r requirements.txt`, then `python -m tests.test_task1 && python -m tests.test_task3`.

## Task 1 – LLM fine-tuning

```bash
python -m task1_ft.data --config configs/task1.yaml                 # select first 40k eligible MetaMathQA rows
python -m task1_ft.diagnostics --config configs/task1.yaml probes   # fix the 32+32 probe set BEFORE any results
python -m task1_ft.evaluate --config configs/task1.yaml --condition baseline
for m in full_ft l2sp lora4 lora16 oplora16; do
  python -m task1_ft.train --config configs/task1.yaml --method $m
done
python -m task1_ft.evaluate --config configs/task1.yaml --condition full_ft --update 625
for m in full_ft l2sp lora4 lora16 oplora16; do python -m task1_ft.evaluate --config configs/task1.yaml --condition $m; done
python -m task1_ft.diagnostics --config configs/task1.yaml cache
python -m task1_ft.diagnostics --config configs/task1.yaml analyze
python -m task1_ft.diagnostics --config configs/task1.yaml verify_oplora
python -m task1_ft.qualitative --config configs/task1.yaml
python -m task1_ft.qualitative --config configs/task1.yaml --annotation-sheet   # 42-row sheet, annotated by hand
python -m task1_ft.summarize --config configs/task1.yaml
```

Implementation notes
* **Selection.** MetaMathQA is scanned in source order; a row is skipped if its query/response is empty, its
  `query` or `original_question` equals a GSM8K test question after NFKC + lower-case + whitespace-collapse
  normalisation, or prompt + response + `<|im_end|>` exceeds 2,048 tokens. IDs and exclusion counts:
  `data/task1/selection.json`. Examples are neither truncated nor shuffled (training follows source order).
* **Formatting / loss.** Prompt = Qwen3 chat template with `enable_thinking=False` (identical to the generation
  prompt); target = response + `<|im_end|>`. CE is summed over all supervised tokens of the 32-example effective
  batch and divided by their count. Logits are formed only for supervised positions, in checkpointed chunks.
* **Precision.** Full FT: FP32 master weights, FP16 autocast, dynamic loss scaling. Adapters: frozen FP16
  backbone, FP32 adapters (the low-rank path itself runs in FP32). Grads are unscaled before clipping; skipped
  updates are logged.
* **L2-SP.** Ω = ½ Σ (θ − θ₀)² over unique parameters (tied embedding counted once), summed, in FP32, added once
  per update with λ = 1e-3. θ₀ is stored on the GPU in bf16 – the checkpoint's own dtype, so the copy is exact
  (checked at start-up; falls back to FP32 if not). SFT loss, Ω, λΩ and total loss are logged separately.
* **LoRA / OPLoRA.** Own implementation (`task1_ft/methods.py`): Kaiming-uniform A, zero B, dropout 0.05, no
  biases, all seven projections. OPLoRA bases come from one FP32 SVD of each W₀; the projected factors
  B̃ = B − U_k(U_kᵀB), Ã = A − (AV_k)V_kᵀ are formed on the fly (no d×d projectors). LoRA-16 and OPLoRA-16 get
  identical initial A, B (same seed, same module order; checked in `tests/test_task1.py`).
* **Evaluation.** lm-evaluation-harness, zero-shot, FP16. GSM8K uses `gsm8k_cot` with the chat template rendered
  with `enable_thinking=False` (small `HFLM` subclass), greedy, `max_gen_toks=1024`; primary metric
  flexible-extract exact match. Adapters are merged into a fresh FP16 backbone (OPLoRA: projected factors) after a
  merged-vs-unmerged FP32 logit check (`results/task1/eval/<cond>/build.json`).
* **Checkpoints.** Resume states at updates 312/625/937/1250 (only the newest is kept to save space); evaluation
  weights at 625 and 1250.

## Task 2 – VLM and visual grounding

```bash
python -m task2_vlm.data download && python -m task2_vlm.data select && python -m task2_vlm.data images
python -m task2_vlm.features --config configs/task2.yaml
python -m task2_vlm.train --config configs/task2.yaml
python -m task2_vlm.evaluate --config configs/task2.yaml
python -m task2_vlm.summarize --config configs/task2.yaml
# optional: --n-vis 49 for train / evaluate / summarize
```

Implementation notes
* **Images** are fetched individually from `images.cocodataset.org` (only the ~15k selected ones). All dev/test
  sets come from val2014 and are mutually image-disjoint; training uses train2014.
* **Features.** `SiglipVisionModel(...).last_hidden_state` = 196 patch tokens (14×14 raster) after SigLIP's
  post-layernorm (applied once, inside HF). Cached once in FP16; the encoder is then freed.
* **Interface.** Visual embeddings are inserted in the user turn before the question. Connector output is scaled
  by a fixed factor = mean text-embedding norm / mean initial connector-output norm, both on training data
  (`results/task2/vis196/calibration.json`). Loss = per-example mean token CE, then mean over the 16 examples.
* **Schedule.** A new AdamW per stage (stage 2 adds LoRA parameters); 5 % warmup, cosine to 10 % of peak.
  Stage 1 = two shuffled passes over 18,000 captions; stages 2–3 draw components with the given probabilities,
  each draw contributing a full question pair or two captions.
* **Evaluation.** Own batched greedy decoder with KV cache (checked against step-by-step decoding). Grounding:
  normalised exact match; VQAv2: local soft accuracy with the official answer normalisation (`vqa_norm.py`).

## Task 3 – Test-time compute

Download the course `ATML-PA3-Data` folder into `data/ATML-PA3-Data/` and MathVista's `testmini.json` into
`data/mathvista_testmini.json`.

```bash
python -m task3_ttc.data inspect        # check column names; edit configs/task3.yaml:columns if needed
python -m task3_ttc.data build          # join + alignment checks + dev/eval split (200/800)
python -m task3_ttc.grading             # answer groups + correctness (math-verify / MathVista rules)
python -m task3_ttc.offline             # Steps 1–2: all selection rules, N in {1..32}, 10 orders, bootstrap
python -m task3_ttc.signals             # Step 3
python -m task3_ttc.adaptive            # Step 5 (thresholds chosen on dev, then frozen)
python -m task3_ttc.live generate       # Step 4 (GPU) ; then: python -m task3_ttc.live summarize
python -m task3_ttc.plots
```

Implementation notes
* Answer groups are formed per question without gold answers (greedy grouping in rollout order); the same
  equivalence rule then grades the groups. Missing extractions form a single `NO_ANSWER` group that votes but is
  always wrong.
* Score and vote ties go to the earliest supporting rollout in the current order; RWC uses average ranks.
  VisConf calibration (5th/95th percentiles, linear interpolation) is recomputed for every candidate prefix.
* Adaptive stopping only sees revealed rollouts; selected thresholds: `results/task3/adaptive/<key>/selected_settings.json`.

## External code / attribution
* `task2_vlm/vqa_norm.py` ports the answer normalisation of the official VQA evaluation script
  (GT-Vision-Lab/VQA, `vqaEval.py`).
* `task3_ttc/grading.py` follows the normalisation rules of MathVista's released scorer and uses
  HuggingFace `math-verify`.
* `data/first_plot_questions.yaml` is copied from the emergent-misalignment repository.
* Evaluation uses EleutherAI's lm-evaluation-harness; models/datasets from the HuggingFace Hub.
* LLM coding assistance was used for parts of the code, as permitted by the PA3 policy.
# 27100150_PA3
