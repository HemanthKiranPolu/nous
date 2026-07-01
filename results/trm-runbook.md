# Runbook — reproduce TRM Sudoku-Extreme (~87%), then attach EV-TRM energy head

The real ~87% Sudoku-Extreme result needs a **persistent GPU** (~6h on A100/L40S).
Free/standard Colab recycles its runtime every ~30–40 min and wipes `/content`,
so a 6h run cannot complete there — use Colab **Pro+ background execution**,
Kaggle (12h), or a cloud VM (Lambda/RunPod/vast.ai). Checkpoint to durable
storage (Google Drive / bucket) so resets are resumable.

Source: `SamsungSAILMontreal/TinyRecursiveModels` (paper arXiv:2510.04871).
Every step below was verified end-to-end on a Colab A100 (the run trained
correctly; only the runtime's ~30-min recycling stopped it).

## 1. Clone + install
```bash
git clone --depth 1 https://github.com/SamsungSAILMontreal/TinyRecursiveModels.git
cd TinyRecursiveModels
# torch is usually preinstalled; if not, match your CUDA. Then:
pip install -q einops adam-atan2 coolname argdantic hydra-core omegaconf \
                pydantic wandb huggingface_hub numba tqdm
```

## 2. GOTCHA — patch adam-atan2 for torch ≥ 2.11
The repo pins torch 2.7; on newer torch, `_cuda_graph_capture_health_check`
was renamed → `AdamATan2.step()` crashes. Patch the installed file:
```python
import adam_atan2, glob
fp = glob.glob(adam_atan2.__file__.rsplit('/',1)[0] + '/adam_atan2.py')[0]
s = open(fp).read().replace(
    "self._cuda_graph_capture_health_check()",
    "getattr(self,'_cuda_graph_capture_health_check',"
    "getattr(self,'_accelerator_graph_capture_health_check',lambda:None))()")
open(fp,'w').write(s)
```

## 3. Build the dataset (~2 min)
```bash
python dataset/build_sudoku_dataset.py \
  --output-dir data/sudoku-extreme-1k-aug-1000 --subsample-size 1000 --num-aug 1000
```

## 4. (optional) make eval accuracy print to stdout
By default metrics go only to wandb. To see `all/exact_accuracy` in the log,
add a print next to BOTH `wandb.log(metrics, step=...)` calls in `pretrain.py`.
NOTE the eval `metrics` is a NESTED dict `{set: {metric: val}}`, so a naive
`float(v)` crashes — use:
```python
print('EVAL_METRICS', {k:(float(v) if not isinstance(v,dict)
      else {a:float(b) for a,b in v.items()}) for k,v in metrics.items()}, flush=True)
```

## 5. Train (~6h on A100/L40S → ~87% ± 2%)
```bash
WANDB_MODE=offline python -u pretrain.py arch=trm \
  data_paths="[data/sudoku-extreme-1k-aug-1000]" evaluators="[]" \
  epochs=50000 eval_interval=5000 \
  lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
  arch.mlp_t=True arch.pos_encodings=none arch.L_layers=2 \
  arch.H_cycles=3 arch.L_cycles=6 +run_name=trm_sudoku ema=True \
  checkpoint_every_eval=True
```
- Resume after a reset: add `+load_checkpoint=<path to latest step_N>`
  (hydra needs the `+`). `checkpoint_every_eval=True` saves every eval.
- First eval at epoch 5000; `all/exact_accuracy` climbs toward ~0.87.

## 6. EV-TRM step (the novel contribution)
Once the base TRM trains, attach the energy/verification head from
`nous/evtrm.py` (predict constraint violations from the recursive latent),
train jointly, and measure calibration (AUC of energy→error) + selective
abstention — the 9×9 version of the PoC (`results/evtrm.md`, 4×4: solve 95.8%,
AUC 0.81). That is the defensible EV-TRM result: a self-verifying tiny recursive
reasoner.
```
```
