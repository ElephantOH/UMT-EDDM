<hr>
<h1 align="center">
  PT-EDDM <br>
  <sub>PT-EDDM: Accurate, Concise, and Efficient Fine-Paired Image Translation</sub>
</h1>
<div align="center">
  <a href="https://github.com/" target="_blank">***</a><sup>1,2</sup> &ensp; <b>&middot;</b> &ensp;
  <a href="https://github.com/" target="_blank">***</a><sup>1,2</sup> &ensp; <b>&middot;</b> &ensp;
  <a href="https://github.com/" target="_blank">***</a><sup>1,2</sup> &ensp; <b>&middot;</b> &ensp;
  <a href="https://github.com/" target="_blank">***</a><sup>1,2</sup> &ensp; <b>&middot;</b> &ensp;
  <a href="https://github.com/" target="_blank">***</a><sup>1,2</sup> &ensp; <b>&middot;</b> &ensp;
  <a href="https://github.com/" target="_blank">***</a><sup>1,2</sup> &ensp; <b>&middot;</b> &ensp;

<span></span>
  <span></span>
  
  <sup>1</sup>111 &emsp; <sup>2</sup>222 &emsp; <sup>3</sup>333 &emsp; <sup>4</sup>444 <br>
</div>
<hr>

<hr>
<h3 align="center">[<a href="https://arxiv.org/">arXiv</a>]</h3>

Official PyTorch implementation of **PT-EDDM**. Experiments demonstrate that our method performs effectively across three medical datasets and one thermal infrared visible light facial dataset.

<p align="center">
  <img src="figures/frame.png" alt="frame" style="width: 1200px; height: auto;">
</p>

## 🐹 Installation

This repository has been developed and tested with `CUDA 11.7` and `Python 3.8`. Below commands create a conda environment with required packages. Make sure conda is installed.

```
conda env create --file requirements.yaml
conda activate eddm
```

## 🐼 Prepare dataset
The default data set class GetDataset requires a specific folder structure for organizing the data set.
Modalities (such as `T1, T2, etc.`) should be stored in separate folders, while splits `(train, test, and optionally val)` should be arranged as subfolders containing `2D` images named `slice_0.png or .npy, slice_1.png or .npy`, and so on.
To utilize your custom data set class, implement your version in `dataset.py` by inheriting from the `EDDMDataset` class.

```
<datasets>/
├── <modality_a>/
│   ├── train/
│   │   ├── slice_0.png or .npy
│   │   ├── slice_1.png or .npy
│   │   └── ...
│   ├── test/
│   │   ├── slice_0.png or .npy
│   │   └── ...
│   └── val/ (The file does not exist by default)
│       ├── slice_0.png or .npy
│       └── ...
├── <modality_b>/
│   ├── train/
│   ├── test/
│   └── val/ (The file does not exist by default)
├── ...
  
```

## 🙉 Training

Execute the following command to start or resume training.
Model checkpoints are stored in the `/checkpoints/$LOG` directory.
The script supports both `single-GPU` and `multi-GPU` training, with `single-GPU` mode being the default setting.

The example training code is as follows: 
```
python train_EDDM.py \
  --input_channels 1 \
  --source T1 \
  --target T2 \
  --batch_size 2 \
  --max_epoch 120 \
  --lr 1.5e-4 \
  --input_path ./datasets/BrainTs20 \
  --checkpoint_path ./checkpoints/brats_1to2_EDDM_logs
```

### Argument descriptions

| Argument                  | Description                                                                                           |
|---------------------------|-------------------------------------------------------------------------------------------------------|
| `--input_channels`        | Dimension of images.                                                                                  |
| `--source` and `--target` | Source Modality and Target Modality, e.g. 'T1', 'T2'. Should match the folder name for that modality. |
| `--batch_size`            | Train set batch size.                                                                                 |
| `--lr`                    | Learning rate.                                                                                        |
| `--max_epoch`             | Number of training epochs (default: 120).                                                             |
| `--input_path`            | Data set directory.                                                                                   |
| `--checkpoint_path`       | Model checkpoint path to resume training.                                                             |
| `--lambda_l1`             | [Optional] Composite ratio of loss items.                                                             |
| `--lambda_l2`             | [Optional] Composite ratio of loss items.                                                             |
| `--lambda_perceptual`     | [Optional] Composite ratio of loss items.                                                             |
| `--vp_t`                  | [Optional] Maximum training time step.                                                                |
| `--val`                   | [Optional] Use the validation phase during the training process.                                      |
| `--MO`                    | [Optional] Hidden state multiplication operator.                                                      |
| `--AO`                    | [Optional] Hidden state addition operator.                                                            |

## 🐣 Testing

Run the following command to start testing.
The predicted images are saved under `/checkpoints/$LOG/generated_samples` directory.
By default, the script runs on a `single GPU`. 

```
python test_EDDM.py \
  --input_channels 1 \
  --source T1 \
  --target T2 \
  --batch_size 1 \
  --which_epoch 120 \
  --gpu_chose 0 \
  --input_path ./datasets/BrainTs20 \
  --checkpoint_path ./checkpoints/brats_1to2_EDDM_logs
```

### Argument descriptions

Some arguments are common to both training and testing and are not listed here. For details on those arguments, please refer to the training section.

| Argument        | Description                                                           |
|-----------------|-----------------------------------------------------------------------|
| `--batch_size`  | Test set batch size.                                                  |
| `--which_epoch` | Model checkpoint path.                                                |
| `--vp_t`        | [Optional] Maximum time step in the inference phase.                  |
| `--vp_max`      | [Optional] Maximum intensity of noise schedule in inference stage.    |
| `--vp_k`        | [Optional] Inference stage noise schedule with added noise curvature. |
| `--vp_sparse`   | [Optional] Sparse coefficient of time step.                           |
| `--vp_noise`    | [Optional] Noise prior mixing ratio.                                  |
| `--vp_prior`    | [Optional] Noise prior mixing ratio.                                  |

## 🐸 Checkpoint

Refer to the testing section above to perform inference with the checkpoints. PSNR (dB), SSIM (%) and MAE are listed as mean ± std across the test set.

| Dataset | Task      | PSNR         | SSIM         | MAE          | Checkpoint                                                   |
|---------|-----------| ------------ | ------------ |--------------| ------------------------------------------------------------ |
| BRATS   | T1→T2     | 31.63 ± 1.53 | 95.64 ± 1.12 | 0.245 ± 1.12 | [Link](https://github.com/) |
| BRATS   | T2→T1     | 31.28 ± 1.56 | 95.03 ± 1.27 | 0.245 ± 1.12 | [Link](https://github.com/) |
| OASIS3  | T1→T2     | 31.63 ± 1.53 | 95.64 ± 1.12 | 0.245 ± 1.12 | [Link](https://github.com/) |
| OASIS3  | T2→T1     | 31.28 ± 1.56 | 95.03 ± 1.27 | 0.245 ± 1.12 | [Link](https://github.com/) |
| IXI     | T1→T2     | 31.63 ± 1.53 | 95.64 ± 1.12 | 0.245 ± 1.12 | [Link](https://github.com/) |
| IXI     | T2→T1     | 31.28 ± 1.56 | 95.03 ± 1.27 | 0.245 ± 1.12 | [Link](https://github.com/) |
| TFW     | VIS.→THE. | 31.63 ± 1.53 | 95.64 ± 1.12 | 0.245 ± 1.12 | [Link](https://github.com/) |

## 🦊 Code

The code for the `test` is open, and the code for the `pre-train` and `tuning` will be made public shortly.

## 🐭 Citation

You are encouraged to modify/distribute this code. However, please acknowledge this code and cite the paper appropriately.
```

```

<hr>
