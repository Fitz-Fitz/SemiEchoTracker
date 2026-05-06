# Semi-Supervised Landmark Tracking in Echocardiography Video via Spatial-Temporal Co-Training and Perception-Aware Attention

Han Wu, Haoyuan Chen, Lin Zhou, Qi Xu, Zhiming Cui, Dinggang Shen  
IEEE Transactions on Medical Imaging, 2026  
DOI: [10.1109/TMI.2026.3651389](https://doi.org/10.1109/TMI.2026.3651389)

## Method Overview

SemiEchoTracker trains with a full echocardiography sequence as input while using landmark supervision only on the first and last frames. The model contains:

- a spatial detector that directly regresses landmark coordinates for every frame;
- a bidirectional temporal tracker initialized from labelled endpoint landmarks;
- endpoint supervision for both detector and tracker;
- temporal tracker regularization with forward/backward consistency and velocity consistency;
- detector-tracker co-training on unlabeled middle frames;
- frozen guided-DINOv2 features with perception-aware spatial-temporal attention.

At inference time, only the detector branch is used to predict landmarks over the whole sequence.

## Code Structure

```text
dataset/echo_dataset.py
models/
  SemiEchoTracker.py
  GCN.py
  agent_attention.py
  dinov2.py
train.py
test.py
```

The guided-DINOv2 pretraining procedure is not included; `models/dinov2.py` expects pretrained DINOv2 weights to be available.

## Setup

Set `SEMI_ECHO_PRETRAIN_DIR` to the pretrained DINOv2 directory:

```bash
export SEMI_ECHO_PRETRAIN_DIR=/path/to/dinov2-finetune
```

The loader expects the hub files and checkpoint in that directory:

```text
$SEMI_ECHO_PRETRAIN_DIR/
  hubconf.py
  teacher_checkpoint.pth
  ...
```

## Training

Semi-supervised training:

```bash
python train.py \
  --data_dir ../data/ \
  --data_type PLAX \
  --training_mode semi
```

Detector-only baseline:

```bash
python train.py \
  --data_dir ../data/ \
  --data_type PLAX \
  --training_mode detector_only
```

## Testing

`test.py` evaluates detector predictions. The checkpoint can be either a run directory or a `.pth` file.

```bash
python test.py \
  --checkpoint ./results/PLAX/YOUR_RUN \
  --data_dir ../data/ \
  --data_type PLAX \
  --split val
```

## Citation

```bibtex
@article{wu2026semiechotracker,
  title={Semi-Supervised Landmark Tracking in Echocardiography Video via Spatial-Temporal Co-Training and Perception-Aware Attention},
  author={Wu, Han and Chen, Haoyuan and Zhou, Lin and Xu, Qi and Cui, Zhiming and Shen, Dinggang},
  journal={IEEE Transactions on Medical Imaging},
  year={2026},
  doi={10.1109/TMI.2026.3651389}
}
```
