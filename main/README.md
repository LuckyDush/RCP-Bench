# RCP-Bench: Robust Collaborative Perception Framework
[CVPR-2025]RCP-Bench: Benchmarking Robustness for Collaborative Perception Under Diverse Corruptions

This repository provides a unified and robustness-oriented multi-agent collaborative perception framework, supporting Global Interference, Ego Interference, CAV Interference!

## 📦 Resources
- 📂 **Datasets**: OPV2V-C, V2XSet-C, DAIR-V2X-C  
- 🛠️ **Toolkit**: Benchmarking scripts & corruption generation  
- 📑 **Paper**: [CVPR 2025 Submission](https://openaccess.thecvf.com/content/CVPR2025/papers/Du_RCP-Bench_Benchmarking_Robustness_for_Collaborative_Perception_Under_Diverse_Corruptions_CVPR_2025_paper.pdf) 


## 📌 Introduction
Collaborative perception enables connected autonomous vehicles (CAVs) to share sensory information, extending perception range and overcoming occlusions. However, **existing studies often assume ideal conditions**, overlooking real-world challenges such as adverse weather, sensor failures, and temporal misalignments.

We present **RCP-Bench**, the first comprehensive benchmark to **evaluate the robustness of collaborative perception models** under diverse corruptions. This work provides:
- Three new datasets: **OPV2V-C**, **V2XSet-C**, and **DAIR-V2X-C**.  
- 14 types of camera corruptions across 6 collaborative scenarios.  
- Extensive evaluation of 10 state-of-the-art collaborative perception models.  
- Two new strategies, **RCP-Drop** and **RCP-Mix**, to improve robustness.  

![RCPBench Teaser](images/rcpbench.pdf)

---

## 🚗 Benchmark Overview
RCP-Bench systematically simulates realistic conditions to test model robustness:

- **Global Interference**: Both ego and CAVs are corrupted.  
- **Ego Interference**: Only ego vehicle is corrupted → collaborative compensation is possible.  
- **CAV Interference**: Only CAVs are corrupted → risk of collaborative disruption.  

### Corruption Categories
- **External Weather**: rain, fog, snow, brightness/darkness, frost.  
- **Camera Interior**: crash, noise (Gaussian, Shot, Impulse), blur (Zoom, Motion, Defocus), quantization.  
- **Temporal Misalignment**: desynchronized capture times.  

Each corruption type is applied at 5 severity levels, resulting in **70 unique corruption conditions**.

---

## 📊 Key Contributions
1. **Benchmarking Robustness**  
   - First large-scale evaluation across **14 corruptions, 3 datasets, and 6 scenarios**.  
   - Metrics: Corrupted AP (AP<sub>cor</sub>), Relative Corruption Error (RCE), Positive Collaborative Coefficient (PosC), and Negative Collaborative Coefficient (NegC).  

2. **New Training Strategies**  
   - **RCP-Drop**: Randomly discard data from collaborating vehicles to simulate sensor/communication failures.  
   - **RCP-Mix**: Mix feature statistics across ego and CAVs to enhance adaptability to distribution shifts.  

3. **Empirical Insights**  
   - Backbone architecture (EfficientNet > ResNet).  
   - Multi-scale fusion improves robustness.  
   - More cameras and CAVs increase stability, but with diminishing returns beyond 4.  
   - Simple fusion methods can outperform complex attention-based ones under corruption.  

---
## 📂 Data Preparation

To get started, you first need to download the original datasets: **OPV2V**, **V2XSet**, and **DAIR-V2X**.

- **OPV2V**: Please refer to the [OpenCOOD repo](https://github.com/DerrickXuNu/OpenCOOD).  
  ⚠️ Additionally, download `additional-001.zip`, which contains the camera modality data.  

- **V2XSet**: Please refer to the [V2X-ViT repo](https://github.com/DerrickXuNu/v2x-vit).  

- **DAIR-V2X**: Download the dataset from the [official page](https://thudair.baai.ac.cn/index).  
  We use the complemented annotation version, so please also follow the instructions provided [here](https://siheng-chen.github.io/dataset/dair-v2x-c-complemented/).  

👉 You may choose to download only the datasets that interest you. However, since **OPV2V** and **DAIR-V2X** are RCPBenchvily used in this repo, we recommend starting with these two.

---

### 📦 Create the RCP-Bench Dataset

Once the original datasets are prepared, you can generate the **corrupted datasets** for evaluation using the provided script.  
⚠️ Note: **We never use corrupted data for training** — only for evaluation. Therefore, all corrupted subsets are generated **from the test split** of OPV2V, V2XSet, and DAIR-V2X.  

```bash
python corrupdataset/dataset.py \
  --root_dir ./data/shihang/RCPBench/test \
  --save_root ./data/shihang/corruptest
  
After processing, the directory structure will look like this:
. 
├── OPV2V
│   ├── additional
│   ├── validate
│   │── train
│   ├── test
│   └── corruptest
│       └── brightness
│           └── 1
│           └── 2
│           └── 3
│           └── 4
│           └── 5
│       └── camera_crash
│           └── 1
│           └── 2
│           └── 3
│           └── 4
│           └── 5
│       └── color_quant
│       └── defocus_blur
│       └── fog
│       └── frost
│       └── gaussian_noise
│       └── impulse_noise
│       └── low_light
│       └── shot_noise
│       └── snow
│       └── zoom_blur
├── V2XSET
│   ├── test
│   ├── train
│   └── validate
│   └── corruptest
└── Dair-V2X
    ├── vehicle-side
    ├── infrastructure-side
    └── cooperative
    └── corruptest
        └── brightness
            └── 1
                └── vehicle-side
                └── infrastructure-side
            └── 2
            └── 3
            └── 4
            └── 5
        └── camera_crash
            └── 1
            └── 2
            └── 3
            └── 4
            └── 5
        └── color_quant
        └── defocus_blur
        └── fog
        └── frost
        └── gaussian_noise
        └── impulse_noise
        └── low_light
        └── shot_noise
        └── snow
        └── zoom_blur
---
##⚙️ Installation

Our installation environment follows CoAlign(https://udtkdfu8mk.feishu.cn/docx/LlMpdu3pNoCS94xxhjMcOWIynie)
 and HEAL(https://github.com/yifanlu0227/HEAL/tree/main).

### Step 1: Basic Installation
```bash
conda create -n rcpbench python=3.8
conda activate rcpbench
# install pytorch. 
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.6 -c pytorch -c conda-forge
# install dependencies (we add matplotlib + imagecorruptions for corruption dataset generation)
pip install -r requirements.txt 
# install this project. It's OK if EasyInstallDeprecationWarning shows up.
python setup.py develop
```

### Step 2: Install Spconv (1.2.1 or 2.x)
To install **spconv 2.x**, check the [table](https://github.com/traveller59/spconv#spconv-spatially-sparse-convolution-library) to run the installation command. For example we have cudatoolkit 11.6, then we should run
```bash
pip install spconv-cu116 # match your cudatoolkit version
```

### Step 3: Bbx IoU cuda version compile
Install bbx nms calculation cuda version
  
```bash
python opencood/utils/setup.py build_ext --inplace
```

### Step 4: Dependencies for FPV-RCNN (optional)
Install the dependencies for fpv-rcnn.
  
```bash
cd RCPBench
python opencood/pcdet_utils/setup.py build_ext --inplace
```
---
##🏋️ Training

1、Training data:
We do not use RCP-Bench corrupted datasets for training — only clean OPV2V, V2XSet, and DAIR-V2X.
Models trained with the HEAL framework can be directly evaluated in RCP-Bench.

When training with our code, set both --type (corruption type) and --level (corruption severity) to None.
```
Example:python opencood/tools/train.py -y ${CONFIG_FILE} [--model_dir ${CHECKPOINT_FOLDER}]
```
-y or hypes_yaml: Path to the training YAML config (e.g. opencood/hypes_yaml/opv2v/LiDAROnly/lidar_fcooper.yaml).

--model_dir (optional): Path to a checkpoint folder for fine-tuning or continued training.


2、RCP-Drop & RCP-Mix:
These are portable modules that can be enabled with minor code changes to improve generalization.

RCP-Drop:
In rcpbench/main/opencood/data_utils/datasets/intermediate_heter_fusion_dataset.py (lines 360–372),
configure the dropout strategy (exponential decay, logarithmic decay, or fixed probability).

dropout_prob=0 → no dropout (baseline)

Higher dropout_prob → more collaborating vehicles are randomly dropped.

RCP-Mix:(for intermediate fusion methods)：
In your YAML config under rcpbench/main/opencood/hypes_yaml/opv2v/CameraOnly/,
change core_method: heter_model_baseline to core_method: rcp_mix.
Example: in camera_v2xvit.yaml, update this field to enable RCP-Mix.
---

##🔬 Evaluation
## 🔧 Configure Dataset Paths
After generating the corrupted datasets, you need to update the corresponding dataset loaders to point to your own `corruptest` directory.

- **For OPV2V / V2XSet**  
  Open:rcpbench/main/opencood/data_utils/dataset/opv2v_basedataset.py
  Locate the following line and replace the hardcoded path with your own:
```python
corcamera_dir = os.path.join("/ssdfs/datahome/tj91066/shihang/corruptest/", str(type), str(level))
Update "/ssdfs/datahome/tj91066/shihang/corruptest/" to the actual path where your corrupted dataset is stored.
For DAIR-V2X
Open:rcpbench/main/opencood/data_utils/datasets/basedataset/dairv2x_basedataset.py
Modify the dataset path at line 160 and line 169 to point to your own corruptest directory.

⚠️ Make sure the modified paths are consistent with the location where you generated the corrupted datasets (e.g., --save_root ./data/shihang/corruptest).

Then Test a model on a specific corruption type and severity on Global Interference:
```
python opencood/tools/inference.py --model_dir ${CHECKPOINT_FOLDER} [--fusion_method intermediate]
```
Fusion methods: single, no, late, early, intermediate (see inference.py for details).

Additional args:type：corruption type
--level  severity: 1-5

Test on all corruptions
```bash
python rcpbench/main/opencood/tools/testcor.py

## 🧪 Testing Ego vs. CAV Interference

To evaluate different interference scenarios, you need to toggle specific lines in the dataset loader:

- **Ego Interference**  
  Open:rcpbench/main/opencood/data_utils/datasets/basedataset/opv2v_basedataset.py
Comment out **lines 281–282** and instead enable **lines 284–285**.

- **CAV Interference**  
In the same file, comment out **lines 275–276** and enable **lines 278–279**.

⚠️ This will switch the dataset behavior to simulate interference on the ego vehicle or on collaborating CAVs.
---


---
🙏 Special thanks to HEAL(https://github.com/yifanlu0227/HEAL/tree/main) for providing the base framework.
---

---
📖 Citation

If you find this repository useful, please cite:

@inproceedings{du2025rcp, title={RCP-Bench: Benchmarking Robustness for Collaborative Perception Under Diverse Corruptions}, author={Du, Shihang and Qu, Sanqing and Wang, Tianhang and Zhang, Xudong and Zhu, Yunwei and Mao, Jian and Lu, Fan and Lin, Qiao and Chen, Guang}, booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference}, pages={11908--11918}, year={2025} }
---