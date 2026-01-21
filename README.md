# DeltaDorsal

Code repository for the paper: **DeltaDorsal: Enhancing Hand Pose Estimation with Dorsal Features in Egocentric Views**


This work is licensed under a
[Creative Commons Attribution 4.0 International License][cc-by].

### [[Project Page]]() [[DOI]]() [[arXiv]]()

![Teaser Image](./assets/teaser.png)

## Overview

DeltaDorsal is a 3D hand pose estimation model using dorsal features of the hand. DeltaDorsal uses a delta encoding approach where features from a base (neutral) hand image are compared with features from the current hand image to predict 3D hand pose and force. Our repo consists of the following components:


- **DeltaDorsalNet**: A backbone network using DINOv3 features with our delta encoding
  - **DINOv3 Backbone**: Extracts features from input images
  - **Change Encoder**: Computes delta features between current and base images
  - **Residual Pose Head**: Predicts pose parameter residuals from prior
  - **Force Head**: Classifies finger activation and force levels
- **Pose Estimation**: Predicts 3D hand pose parameters
- **Force Estimation**: Classifies finger activation and force levels

## Installation

### Prerequisites

- Python >= 3.10
- [uv](https://github.com/astral-sh/uv) or any other package manager of your choice

### Setup

1. Clone the repository:

```bash
git clone --recursive [REPO_URL]
cd wrinklesense
```

2. Create a virtual environment and install PyTorch:

```bash
uv pip install torch --torch-backend=auto
```

3. Install all dependencies:

```bash
uv sync
```

4. Download DINOv3

Make sure to follow all the installation instructions for DINOv3 found [here](https://github.com/facebookresearch/dinov3).

Download all model weights and place them into ```_DATA/dinov3```

5. Download MANO

Our repo requires the usage of the MANO hand model. Request access [here](https://mano.is.tue.mpg.de/).

You only need to put ```MANO_RIGHT.pkl``` under the  ```_DATA/mano``` folder.

6. (Optional) Set up Weights & Biases for logging:

```bash
wandb login
```

## Data Preparation

We organize our data as follows according to the following setup which can interface with our prewritten dataset modules found in ```src/datasets/```. If you want to use your own data, please feel free to write your own modules.

```
.                                         # ROOT
├── bases.json                            # Bases metadata
├── trials.json                           # Metadata of each trial
├── frames.json                           # Metadata of all captured frames
├── train.json                            # (OPTIONAL) subset of frames.json for train split
├── val.json                              # (OPTIONAL) subset of frames.json for val split
├── test.json                             # (OPTIONAL) subset of frames.json for test
├── trials/                               # all captured data
│   ├── PARTICIPANT_XXX/
│   │   ├── TRIAL_XXX/
│   │   │   ├── anns/
│   │   │   │   ├── frame_XXX.npy
│   │   │   │   ├── frame_XXX.npy
│   │   │   │   └── frame_XXX.npy
│   │   │   ├── hamer/                    # initial pose prediction
│   │   │   │   ├── frame_XXX.npy
│   │   │   │   ├── frame_XXX.npy
│   │   │   │   └── frame_XXX.npy
│   │   │   ├── imgs/                     # captured images
│   │   │   │   ├── frame_XXX.jpg
│   │   │   │   ├── frame_XXX.jpg
│   │   │   │   └── frame_XXX.jpg
│   │   │   ├── cropped_images/           # (OPTIONAL) Precropped images that are aligned to bases
│   │   │   │   ├── frame_XXX.jpg
│   │   │   │   ├── frame_XXX.jpg
│   │   │   │   └── frame_XXX.jpg
│   │   │   └── cropped_bases/            # (OPTIONAL) Precropped bases that are aligned to each frame
│   │   │       ├── frame_XXX.jpg
│   │   │       ├── frame_XXX.jpg
│   │   │       └── frame_XXX.jpg
│   │   └── ...
│   └── ...
└── bases/                               # all captured reference images
    ├── PARTICIPANT_XXX/
    │   ├── hamer/                       # initial pose prediction
    │   │   ├── frame_XXX.npy
    │   │   ├── frame_XXX.npy
    │   │   └── frame_XXX.npy
    │   └── imgs/
    │       ├── frame_XXX.jpg
    │       ├── frame_XXX.jpg
    │       └── frame_XXX.jpg
    └── ...
```

Each frame should have:
- Image file
- HaMeR pose predictions or some initial 2d pose prediction for alignment
- Ground truth pose annotations
- Force sensor readings for force estimation (optional)

### Data Schemas

```base.json```

- bases_dir (str) - path to bases dir (default to "bases")
- participants (array)
  - item (object)
    - p_id (int) - participant id
    - bases (array)
      - item (object)
        - base_id (int) - base id
        - img_path (str) - relative path from bases_dir to the base image .jpg
        - hamer_path (str) - relative path from bases_dir to the initial annotations .npy


```trials.json```

- trials (array)
  - item (object)
    - trial_id (int) - id of this trial
    - p_id (int) - participant id for this trial
    - motion_type (str) - type of gesture
    - hand_position (str) - orientation of hand
    - K (array(float)) - 3x3 camera intrinsic matrix (Represented as A [here](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html))
    - d (array(float)) - 1x15 camera intrinsic distortion (output from OpenCV)
    - world2cam (array(float)) - 4x4 camera extrinsic matrix.

```frames.json, train.json, val.json, test.json```

- frames_dir (str) - path to frames dir (default to "trials")
- split (str) - training split. one of full, train, val, test
- frames (array)
  - item (object)
    - trial_id (int) - id of corresponding trial
    - timestamp (float) - timestamp within the video capture
    - frame_no (int) - index of frame in the video
    - img_path (str) - relative path from frames_dir to the captured image .jpg
    - cropped_img_path (str) - (OPTIONAL) relative path from frames_dir to a precropped and aligned image .jpg
    - cropped_base_path (str) - (OPTIONAL) relative path from frames_dir to a precropped and aligned reference image .jpg
    - cropped_base_idx (int) - index of base image taken and prealigned for this frame
    - ann_path (str) - relative path from frames_dir to the ground truth annotation .npy
    - hamer_path (str) - relative path from frames_dir to the initial annotations .npy
    - fsr_reading (float) - (OPTIONAL) fsr reading for this frame for force predictions
    - tap_label (float) - (OPTIONAL) assigned label for force action type
    - frame_id (int) - id for this frame data


```annotation.npy``` (all annotation files)

- betas (array) - 1x10 shape parameters for MANO
- global_orient (array) - 1x3 global orientation for MANO (commonly represented as the first three terms of pose parameters)
- hand_pose (array) - 1x15x3 pose parameters in axis-angle representation for MANO. (Can be 1x15x3x3 if in rotation matrix form)
- cam_t (array) - 1x3 camera projection parameters to convert from 3D to 2D camera frame annotations
- keypoints_3d (array) - 21x3 openpose keypoint locations in xyz format
- keypoints_2d (array) - 21x2 openpose keypoint locations projected into camera frame



### Face Label Mapping

For processing, finger labels are mapped as follows:
- 0: index
- 1: middle
- 2: ring
- 3: pinky
- 4: thumb
- 5: dorsal
- 6: palm
- 7: other

where the assignment of mano face labels to each finger label can be found in [```assets/mano_face_labels.json```](./assets/mano_face_labels.json).

## Training

DeltaDorsal uses [Hydra](https://hydra.cc/) for configuration management. Config files are located in [```configs/```](./configs/):

### Modifying Configs

You can override any config parameter via command line:

```bash
python src/train_pose.py model.optimizer.lr=0.0001 data.batch_size=32 max_epochs=100
```

Or create a new config file in `configs/experiments/` and reference it:

```bash
python src/train_pose.py experiment=my_experiment
```


### Pose Model Training

Train the pose estimation model:

```bash
python src/train_pose.py
```

#### Common Configuration Overrides

Override config parameters via command line:

```bash
# Change number of epochs
python src/train_pose.py max_epochs=50

# Change batch size
python src/train_pose.py data.batch_size=32

# Change learning rate
python src/train_pose.py model.optimizer.lr=1e-4

# Resume from checkpoint
python src/train_pose.py ckpt_path=path/to/checkpoint.ckpt

# Train only (skip testing)
python src/train_pose.py test=False

# Test only (skip training)
python src/train_pose.py train=False test=True ckpt_path=path/to/checkpoint.ckpt
```

#### Training with Different Data Splits

```bash
# Use leave-one-out cross-validation
python src/train_pose.py data.leave_one_out=True data.val_participants=[1] data.test_participants=[3]

# Train on specific participants
python src/train_pose.py data.all_participant_ids=[1,2,3,4,5]
```

### Force Model Training

Train the force estimation model:

```bash
python src/train_force.py
```

#### Common Configuration Overrides

```bash
# Change number of epochs
python src/train_force.py max_epochs=50

# Change batch size
python src/train_force.py data.batch_size=64

# Resume from checkpoint
python src/train_force.py ckpt_path=path/to/checkpoint.ckpt

# Train only
python src/train_force.py test=False

# Test only
python src/train_force.py train=False test=True ckpt_path=path/to/checkpoint.ckpt
```

## Inference and Testing

### Running Test Set Evaluation

After training, evaluate on the test set:

```bash
# Pose model evaluation
python src/train_pose.py train=False test=True ckpt_path=path/to/checkpoint.ckpt

# Force model evaluation
python src/train_force.py train=False test=True ckpt_path=path/to/checkpoint.ckpt
```

The checkpoint path can be:
- Explicit path: `ckpt_path=outputs/2024-01-01_12-00-00/checkpoints/best.ckpt`
- Best checkpoint from training: If `ckpt_path` is not specified and training was run, it will use the best checkpoint automatically

### Test Outputs

Test results are saved to the Hydra output directory (typically `outputs/YYYY-MM-DD_HH-MM-SS/`):

- **Pose model**: `test_outputs_epoch{N}.npy` containing:
  - frame_id (int) - frame identifiers
  - pose (array) - predicted pose parameters
  - shape (array) - hand shape parameters
  - pred_kp_3d (array): predicted 3D keypoints
  - gt_kp_3d (array): ground truth 3D keypoints
  - hamer_kp_3d (array): prior (HaMeR) 3D keypoints

- **Force model**: `test_outputs_epoch{N}.npy` containing:
  - frame_id (int) - frame identifiers
  - pred_tap_labels (int) - Predicted force labels
  - gt_tap_labels (int) - Ground truth force labels

### Metrics

During validation and testing, the following metrics are logged:

**Pose Model:**
- `val/mpjpe`: Mean Per Joint Position Error (mm)
- `val/mpjpe-prior`: MPJPE of the prior (HaMeR) predictions
- `val/loss`: Total loss (keypoint + pose parameter loss)
- `val/val-inference-time`: Inference time per sample (ms)

**Force Model:**
- `val/acc`: Classification accuracy
- `val/prec_w`: Weighted precision
- `val/recall_w`: Weighted recall
- `val/f1_w`: Weighted F1 score
- `val/loss`: Cross-entropy loss
- `val/inference-time`: Inference time per sample (ms)

### Key Configuration Parameters

**Training:**
- `max_epochs`: Maximum training epochs
- `min_epochs`: Minimum training epochs
- `train`: Whether to run training
- `test`: Whether to run testing
- `ckpt_path`: Path to checkpoint for resuming/testing
- `seed`: Random seed

**Data:**
- `data.data_dir`: Path to dataset directory
- `data.batch_size`: Batch size per device
- `data.out_img_size`: Input image size
- `data.leave_one_out`: Use leave-one-out cross-validation
- `data.val_participants`: Participant IDs for validation
- `data.test_participants`: Participant IDs for testing

**Model:**
- `model.backbone.model_name`: DINOv3 model variant
- `model.backbone.n_unfrozen_blocks`: Number of unfrozen transformer blocks
- `model.optimizer.lr`: Learning rate
- `model.scheduler`: Learning rate scheduler config

## Model Architecture


The model uses a residual prediction approach where it predicts corrections to a prior pose estimate (from HaMeR), enabling more stable training and better generalization.

## Acknowledgements

Parts of the code are taken or adapted from the following repositories:

- [4DHumans](https://github.com/shubham-goel/4D-Humans)
- [SLAHMR](https://github.com/vye16/slahmr)
- [ProHMR](https://github.com/nkolot/ProHMR)
- [SPIN](https://github.com/nkolot/SPIN)
- [SMPLify-X](https://github.com/vchoutas/smplify-x)
- [HMR](https://github.com/akanazawa/hmr)
- [ViTPose](https://github.com/ViTAE-Transformer/ViTPose)
- [Detectron2](https://github.com/facebookresearch/detectron2)
- [HaMeR](https://github.com/geopavlakos/hamer)
- [DINOv3](https://github.com/facebookresearch/dinov3)

## Citing

If extending or using our work, please cite the following papers:

```

PLACEHOLDER

@article{MANO:SIGGRAPHASIA:2017,
  title = {Embodied Hands: Modeling and Capturing Hands and Bodies Together},
  author = {Romero, Javier and Tzionas, Dimitrios and Black, Michael J.},
  journal = {ACM Transactions on Graphics, (Proc. SIGGRAPH Asia)},
  publisher = {ACM},
  month = nov,
  year = {2017},
  url = {http://doi.acm.org/10.1145/3130800.3130883},
  month_numeric = {11}
}

@misc{simeoniDINOv32025,
  title = {{{DINOv3}}},
  author = {Sim{\'e}oni, Oriane and Vo, Huy V. and Seitzer, Maximilian and Baldassarre, Federico and Oquab, Maxime and Jose, Cijo and Khalidov, Vasil and Szafraniec, Marc and Yi, Seungeun and Ramamonjisoa, Micha{\"e}l and Massa, Francisco and Haziza, Daniel and Wehrstedt, Luca and Wang, Jianyuan and Darcet, Timoth{\'e}e and Moutakanni, Th{\'e}o and Sentana, Leonel and Roberts, Claire and Vedaldi, Andrea and Tolan, Jamie and Brandt, John and Couprie, Camille and Mairal, Julien and J{\'e}gou, Herv{\'e} and Labatut, Patrick and Bojanowski, Piotr},
  year = 2025,
  month = aug,
  number = {arXiv:2508.10104},
  eprint = {2508.10104},
  primaryclass = {cs},
  publisher = {arXiv},
  doi = {10.48550/arXiv.2508.10104},
  urldate = {2025-08-25},
  archiveprefix = {arXiv}
}


```
