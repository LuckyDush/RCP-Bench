import os
cor_type=['brightness','cameracrash','color_quant','defocus_blur','fog','frost','gaussian_noise','impulse_noise','low_light','motion_blur','shot_noise', 'snow', 'zoom_blur']

for corruption in cor_type:
        for severity in range(1,6):
                cmd = f"python opencood/tools/inferencenorm.py --model_dir /share/home/tj91066/data/shihang/HEAL/opencood/logs1/v2xvit0.1 --fusion_method intermediate --type {corruption} --level {severity}"
                print(f"Running command: {cmd}")
                os.system(cmd)


for corruption in cor_type:
        for severity in range(1,6):
                cmd = f"python opencood/tools/inferencenorm.py --model_dir /share/home/tj91066/data/shihang/HEAL/opencood/logs1/v2xvitlinear --fusion_method intermediate --type {corruption} --level {severity}"
                print(f"Running command: {cmd}")
                os.system(cmd)