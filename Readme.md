# RCP-Bench: Robust Collaborative Perception Framework
[CVPR-2025] **RCP-Bench: Benchmarking Robustness for Collaborative Perception Under Diverse Corruptions**

This repository provides a unified and robustness-oriented multi-agent collaborative perception framework, supporting **Global Interference**, **Ego Interference**, and **CAV Interference**.

---

## 📦 Resources
- 📂 **Datasets**: OPV2V-C, V2XSet-C, DAIR-V2X-C  
- 🛠️ **Toolkit**: Benchmarking scripts & corruption generation  
- 📑 **Paper**: [CVPR 2025 Submission](https://openaccess.thecvf.com/content/CVPR2025/papers/Du_RCP-Bench_Benchmarking_Robustness_for_Collaborative_Perception_Under_Diverse_Corruptions_CVPR_2025_paper.pdf)  

---

## 📌 Introduction
Collaborative perception enables connected autonomous vehicles (CAVs) to share sensory information, extending perception range and overcoming occlusions. However, **existing studies often assume ideal conditions**, overlooking real-world challenges such as adverse weather, sensor failures, and temporal misalignments.

We present **RCP-Bench**, the first comprehensive benchmark to **evaluate the robustness of collaborative perception models** under diverse corruptions. This work provides:
- Three new datasets: **OPV2V-C**, **V2XSet-C**, and **DAIR-V2X-C**  
- 14 types of camera corruptions across 6 collaborative scenarios  
- Extensive evaluation of 10 state-of-the-art collaborative perception models  
- Two new strategies, **RCP-Drop** and **RCP-Mix**, to improve robustness  

![RCPBench Teaser](images/rcpbench.pdf)

---

## 🚗 Benchmark Overview
RCP-Bench systematically simulates realistic conditions to test model robustness:

- **Global Interference**: Both ego and CAVs are corrupted  
- **Ego Interference**: Only ego vehicle is corrupted → collaborative compensation is possible  
- **CAV Interference**: Only CAVs are corrupted → risk of collaborative disruption  

### Corruption Categories
- **External Weather**: rain, fog, snow, brightness/darkness, frost  
- **Camera Interior**: crash, noise (Gaussian, Shot, Impulse), blur (Zoom, Motion, Defocus), quantization  
- **Temporal Misalignment**: desynchronized capture times  

Each corruption type is applied at 5 severity levels, resulting in **70 unique corruption conditions**.

---

## 📊 Key Contributions
1. **Benchmarking Robustness**  
   - First large-scale evaluation across **14 corruptions, 3 datasets, and 6 scenarios**  
   - Metrics: Corrupted AP (AP<sub>cor</sub>), Relative Corruption Error (RCE), Positive Collaborative Coefficient (PosC), and Negative Collaborative Coefficient (NegC)  

2. **New Training Strategies**  
   - **RCP-Drop**: Randomly discard data from collaborating vehicles to simulate sensor/communication failures  
   - **RCP-Mix**: Mix feature statistics across ego and CAVs to enhance adaptability to distribution shifts  

3. **Empirical Insights**  
   - Backbone architecture (EfficientNet > ResNet)  
   - Multi-scale fusion improves robustness  
   - More cameras and CAVs increase stability, but diminishing returns beyond 4  
   - Simple fusion methods can outperform complex attention-based ones under corruption  

---

## 📂 Data Preparation

To get started, you first need to download the original datasets: **OPV2V**, **V2XSet**, and **DAIR-V2X**.

- **OPV2V**: Please refer to the [OpenCOOD repo](https://github.com/DerrickXuNu/OpenCOOD)  
  ⚠️ Additionally, download `additional-001.zip`, which contains the camera modality data  

- **V2XSet**: Please refer to the [V2X-ViT repo](https://github.com/DerrickXuNu/v2x-vit)  

- **DAIR-V2X**: Download the dataset from the [official page](https://thudair.baai.ac.cn/index)  
  We use the complemented annotation version, so please also follow the instructions provided [here](https://siheng-chen.github.io/dataset/dair-v2x-c-complemented/)  

👉 You may choose to download only the datasets that interest you. However, since **OPV2V** and **DAIR-V2X** are heavily used in this repo, we recommend starting with these two.

---

### 📦 Create the RCP-Bench Dataset
Once the original datasets are prepared, you can generate the **corrupted datasets** for evaluation using the provided script.  

⚠️ Note: **We never use corrupted data for training** — only for evaluation. Therefore, all corrupted subsets are generated **from the test split** of OPV2V, V2XSet, and DAIR-V2X.  

```bash
python corrupdataset/dataset.py \
  --root_dir ./data/shihang/RCPBench/test \
  --save_root ./data/shihang/corruptest
After processing, the directory structure will look like this:

bash

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
⚙️ Installation
Our installation environment follows CoAlign and HEAL.

Step 1: Basic Installation
bash

conda create -n rcpbench python=3.8
conda activate rcpbench

# install pytorch
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.6 -c pytorch -c conda-forge

# install dependencies (add matplotlib + imagecorruptions for corruption dataset generation)
pip install -r requirements.txt 

# install this project (OK if EasyInstallDeprecationWarning shows up)
python setup.py develop
Step 2: Install Spconv (1.2.1 or 2.x)
Check the table to match your CUDA version. Example for CUDA 11.6:

bash

pip install spconv-cu116
Step 3: Compile Bbx IoU (CUDA version)
bash

python opencood/utils/setup.py build_ext --inplace
Step 4: Dependencies for FPV-RCNN (optional)
bash

cd RCPBench
python opencood/pcdet_utils/setup.py build_ext --inplace
🏋️ Training
Training Data
We do not use RCP-Bench corrupted datasets for training — only clean OPV2V, V2XSet, and DAIR-V2X.
Models trained with the HEAL framework can be directly evaluated in RCP-Bench.

bash

python opencood/tools/train.py -y ${CONFIG_FILE} [--model_dir ${CHECKPOINT_FOLDER}]
-y: Path to YAML config (e.g. opencood/hypes_yaml/opv2v/LiDAROnly/lidar_fcooper.yaml)

--model_dir (optional): Path to checkpoint folder

RCP-Drop & RCP-Mix

RCP-Drop: Configure dropout strategy in intermediate_heter_fusion_dataset.py (lines 360–372)

RCP-Mix: Change core_method in YAML config from heter_model_baseline to rcp_mix

🔬 Evaluation
🔧 Configure Dataset Paths
Update dataset loaders to point to your corruptest directory.

For OPV2V / V2XSet: Modify paths in opv2v_basedataset.py

For DAIR-V2X: Modify paths at line 160 & 169 in dairv2x_basedataset.py

Example:

python

corcamera_dir = os.path.join("/your/path/to/corruptest/", str(type), str(level))
Run Evaluation
Single corruption:

bash

python opencood/tools/inference.py --model_dir ${CHECKPOINT_FOLDER} [--fusion_method intermediate]
All corruptions:

bash

python rcpbench/main/opencood/tools/testcor.py
🧪 Testing Ego vs. CAV Interference
Modify dataset loader:

Ego Interference: Comment out lines 281–282, enable 284–285

CAV Interference: Comment out lines 275–276, enable 278–279

🙏 Acknowledgements
Special thanks to HEAL for providing the base framework.

📖 Citation
If you find this repository useful, please cite:
@inproceedings{du2025rcp, title={RCP-Bench: Benchmarking Robustness for Collaborative Perception Under Diverse Corruptions}, author={Du, Shihang and Qu, Sanqing and Wang, Tianhang and Zhang, Xudong and Zhu, Yunwei and Mao, Jian and Lu, Fan and Lin, Qiao and Chen, Guang}, booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference}, pages={11908--11918}, year={2025} }
