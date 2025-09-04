import os
for severity in range(1,6):
        cmd = f"python opencood/tools/inference.py --model_dir /share/home/tj91066/data/shihang/HEAL/opencood/logstestcluver/HeterBaseline_opv2v_camera_attfuse_2024_09_15_23_42_41  --fusion_method intermediate --type tempormisalign --level {severity}"
        print(f"Running command: {cmd}")
        os.system(cmd)

for severity in range(1,6):
        cmd = f"python opencood/tools/inference.py --model_dir /share/home/tj91066/data/shihang/HEAL/opencood/logstestcluver/HeterBaseline_opv2v_camera_cobevt_2024_09_25_16_04_22  --fusion_method intermediate --type tempormisalign --level {severity}"
        print(f"Running command: {cmd}")
        os.system(cmd)

for severity in range(1,6):
        cmd = f"python opencood/tools/inference.py --model_dir /share/home/tj91066/data/shihang/HEAL/opencood/logstestcluver/HeterBaseline_opv2v_camera_disco_2024_09_14_14_37_41  --fusion_method intermediate --type tempormisalign --level {severity}"
        print(f"Running command: {cmd}")
        os.system(cmd)

for severity in range(1,6):
        cmd = f"python opencood/tools/inference.py --model_dir /share/home/tj91066/data/shihang/HEAL/opencood/logstestcluver/HeterBaseline_opv2v_camera_fcooper_2024_09_13_12_23_39 --fusion_method intermediate --type tempormisalign --level {severity}"
        print(f"Running command: {cmd}")
        os.system(cmd)

for severity in range(1,6):
        cmd = f"python opencood/tools/inference.py --model_dir /share/home/tj91066/data/shihang/HEAL/opencood/logstestcluver/HeterBaseline_opv2v_camera_max_2024_09_26_11_00_19 --fusion_method intermediate --type tempormisalign --level {severity}"
        print(f"Running command: {cmd}")
        os.system(cmd)


for severity in range(1,6):
        cmd = f"python opencood/tools/inference.py --model_dir /share/home/tj91066/data/shihang/HEAL/opencood/logstestcluver/HeterBaseline_opv2v_camera_v2vnet_2024_09_12_08_10_15 --fusion_method intermediate --type tempormisalign --level {severity}"
        print(f"Running command: {cmd}")
        os.system(cmd)


for severity in range(1,6):
        cmd = f"python opencood/tools/inference.py --model_dir /share/home/tj91066/data/shihang/HEAL/opencood/logstestcluver/HeterBaseline_opv2v_camera_v2xvit_2024_09_11_03_10_22 --fusion_method intermediate --type tempormisalign --level {severity}"
        print(f"Running command: {cmd}")
        os.system(cmd)

for severity in range(1,6):
        cmd = f"python opencood/tools/inference.py --model_dir /share/home/tj91066/data/shihang/HEAL/opencood/logstestcluver/late --fusion_method intermediate --type tempormisalign --level {severity}"
        print(f"Running command: {cmd}")
        os.system(cmd)