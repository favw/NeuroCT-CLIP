# NeuroCT-CLIP

NeuroCT-CLIP adapts CT-CLIP for image-text contrastive learning on head CT volumes and radiology reports. 

Original repository: https://github.com/ibrahimethemhamamci/CT-CLIP

Original paper: https://arxiv.org/abs/2403.17834

## Training

From the `scripts` directory:

```bash
python run_train.py \
  --head \
  --train-data-folder /path/to/train/volumes \
  --valid-data-folder /path/to/valid/volumes \
  --train-reports-file /path/to/train_reports.csv \
  --valid-reports-file /path/to/valid_reports.csv \
  --labels /path/to/labels.csv \
  --results-folder /path/to/output \
  --batch-size 8 \
  --target-depth 90 \
  --num-train-steps 20000 \
  --save-every 1000 \
  --preprocessed-nifti
```

To continue from a checkpoint, add:

```bash
--pretrained /path/to/checkpoint.pt
```

## Inference

```bash
python run_zero_shot.py \
  --head \
  --pretrained /path/to/checkpoint.pt \
  --data-folder /path/to/volumes \
  --reports-file /path/to/reports.csv \
  --labels /path/to/labels.csv \
  --save /path/to/output \
  --target-depth 90 \
  --preprocessed-nifti
```

`--preprocessed-nifti` should only be used for already preprocessed NIfTI volumes.
