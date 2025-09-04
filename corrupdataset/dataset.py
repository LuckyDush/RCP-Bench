import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import PIL
import PIL.ImageOps
from imagecorruptions import corrupt

# -------------------------------
# Common Utility
# -------------------------------
def extract_timestamps(yaml_files):
    """Extract mocked timestamps from YAML filenames."""
    timestamps = []
    for file in yaml_files:
        res = file.split('/')[-1]
        timestamp = res.replace('.yaml', '')
        timestamps.append(timestamp)
    return timestamps

def save_image(img, save_path):
    plt.imshow(img)
    plt.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(save_path)
    plt.close()

# -------------------------------
# Corruption Functions
# -------------------------------
def apply_noise_blur(img, corruption_name, severity):
    return corrupt(img, corruption_name=corruption_name, severity=severity)

def apply_color_quant(img, severity):
    bits = 5 - severity
    img = Image.fromarray(np.uint8(img))
    img = PIL.ImageOps.posterize(img, bits)
    return np.asarray(img)

def imadjust(x, a, b, c, d, gamma=1):
    y = (((x - a) / (b - a)) ** gamma) * (d - c) + c
    return y

def poisson_gaussian_noise(x, severity):
    c_poisson = 10 * [60, 25, 12, 5, 3][severity]
    x = np.array(x) / 255.
    x = np.clip(np.random.poisson(x * c_poisson) / c_poisson, 0, 1) * 255
    c_gauss = 0.1 * [.08, .12, 0.18, 0.26, 0.38][severity]
    x = np.array(x) / 255.
    x = np.clip(x + np.random.normal(size=x.shape, scale=c_gauss), 0, 1) * 255
    return np.uint8(x)

def apply_low_light(img, severity):
    c = [0.60, 0.50, 0.40, 0.30, 0.20][severity]
    x = np.array(img) / 255.
    x_scaled = imadjust(x, x.min(), x.max(), 0, c, gamma=2.) * 255
    return poisson_gaussian_noise(x_scaled, severity=severity)

def apply_camera_crash(images, severity):
    """Randomly zero-out N cameras to simulate crash."""
    mics = np.unique(np.random.choice([0, 1, 2, 3], severity+1))
    for m in mics:
        images[m] = np.zeros((600, 800, 3))
    return images

# -------------------------------
# Main Processing
# -------------------------------
def process_folder(corruption, severity, root_dir, save_root):
    scenario_folders = sorted([os.path.join(root_dir, x)
                               for x in os.listdir(root_dir)
                               if os.path.isdir(os.path.join(root_dir, x))])
    for scenario_folder in scenario_folders:
        cav_list = sorted([x for x in os.listdir(scenario_folder)
                           if os.path.isdir(os.path.join(scenario_folder, x))])
        for cav_id in cav_list:
            cav_path = os.path.join(scenario_folder, cav_id)
            yaml_files = sorted([os.path.join(cav_path, x)
                                 for x in os.listdir(cav_path)
                                 if x.endswith('.yaml') and 'additional' not in x])
            timestamps = extract_timestamps(yaml_files)

            for timestamp in timestamps:
                path = f"{corruption}/{severity+1}"
                camera_new_path = cav_path.replace(root_dir, f"{save_root}/{path}")
                camera_new_path1 = cav_path.replace(root_dir, f"{path}")
                os.makedirs(camera_new_path, exist_ok=True)

                # Skip if already exists
                if os.path.exists(os.path.join(camera_new_path, timestamp + '_camera3.png')):
                    continue

                # Load original images
                imgs = []
                for cam in range(4):
                    img_file = os.path.join(cav_path, f"{timestamp}_camera{cam}.png")
                    imgs.append(np.asarray(Image.open(img_file)))

                # Apply corruption
                if corruption in ['gaussian_noise', 'shot_noise', 'impulse_noise',
                                  'defocus_blur', 'motion_blur', 'zoom_blur',
                                  'snow', 'frost', 'fog', 'brightness']:
                    for i in range(4):
                        corrupted = apply_noise_blur(imgs[i], corruption, severity+1)
                        save_image(corrupted, f"{save_root}/{camera_new_path1}/{timestamp}_camera{i}.png")

                elif corruption == 'color_quant':
                    for i in range(4):
                        corrupted = apply_color_quant(imgs[i], severity)
                        save_image(corrupted, f"{save_root}/{camera_new_path1}/{timestamp}_camera{i}.png")

                elif corruption == 'low_light':
                    for i in range(4):
                        corrupted = apply_low_light(imgs[i], severity)
                        save_image(corrupted, f"{save_root}/{camera_new_path1}/{timestamp}_camera{i}.png")

                elif corruption == 'camera_crash':
                    crashed_imgs = apply_camera_crash(imgs, severity)
                    for i in range(4):
                        save_image(crashed_imgs[i], f"{save_root}/{camera_new_path1}/{timestamp}_camera{i}.png")

# -------------------------------
# Entry Point
# -------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply image corruptions for collaborative perception dataset.")
    parser.add_argument("--root_dir", type=str, required=True, help="Path to the source dataset root directory.")
    parser.add_argument("--save_root", type=str, required=True, help="Path to save the corrupted dataset.")
    args = parser.parse_args()

    items = ['gaussian_noise', 'shot_noise', 'impulse_noise',
             'defocus_blur', 'motion_blur', 'zoom_blur',
             'snow', 'frost', 'fog', 'brightness',
             'color_quant', 'low_light', 'camera_crash']
    severity_levels = range(5)

    for corruption in items:
        for severity in severity_levels:
            print(f"Processing {corruption} severity {severity+1}")
            process_folder(corruption, severity, args.root_dir, args.save_root)
