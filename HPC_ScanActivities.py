# Author(s): Dr. Patrick Lemoine

import os
import sys
import numpy as np
import psutil
import cv2
import time
import logging
from datetime import datetime
import psutil
import matplotlib.pyplot as plt

def setup_logger(path):
    log_filename = datetime.now().strftime("hpc_cpu_activity_%Y-%m-%d_%H-%M-%S.log")
    log_filename = path+"/"+log_filename
    
    logging.basicConfig(
        filename=log_filename,
        filemode='w',
        format='%(asctime)s - %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logger = logging.getLogger()
    return logger


def run_level1(path, interval, num_samples):
    num_cpus = psutil.cpu_count(logical=True)
    activity_matrix = np.zeros((num_cpus, num_samples))
    print(f"Monitoring of {num_cpus} CPU(s), {num_samples} intervals...")
    for t in range(num_samples):
        usages = psutil.cpu_percent(interval=interval, percpu=True)
        activity_matrix[:, t] = usages
        print(f"Sample {t+1}: {usages}")
        
    plt.figure(figsize=(14, 6))
    plt.imshow(activity_matrix, aspect='auto', cmap='RdYlBu_r', vmin=0, vmax=100)
    plt.colorbar(label='CPU Usage (%)')
    plt.xlabel("Time (seconds)")
    plt.ylabel("CPU Number")
    plt.title("CPU Activity Monitoring Matrix")
    
    output_file = datetime.now().strftime("CPU_Activity_Monitoring_Matrix_%Y-%m-%d_%H-%M-%S.png")
    output_file = path+"/"+output_file
    plt.savefig(output_file)
    plt.show()
    

def run_level2(path, interval, height):
    num_cpus = psutil.cpu_count(logical=True)    
    width = num_cpus
    matrix = np.zeros((height, width), dtype=np.float32)
    cv2.setUseOptimized(True)

    logger = setup_logger(path)
    logger.info(f"Monitoring CPU activity for {num_cpus} CPUs with interval {interval}s")

    last_time = time.time()
    frame_count = 0
    
    nameWindow = "HPC "+str(num_cpus)+" CPUs Scan Activities "
    cv2.namedWindow(nameWindow, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(nameWindow, num_cpus*25, 800)

    try:
        while True:
            start_loop = time.time()

            activity = np.zeros(width, dtype=np.float32)
            activity[:num_cpus] = psutil.cpu_percent(percpu=True)

            matrix[1:] = matrix[:-1]
            matrix[0] = activity

            logger.info(" | ".join(f"CPU{i}: {activity[i]:.1f}%" for i in range(num_cpus)))

            img_uint8 = np.clip(matrix * 255 / 100, 0, 255).astype(np.uint8)
            color_img = cv2.applyColorMap(img_uint8, cv2.COLORMAP_JET)
            fps = frame_count / (time.time() - last_time + 1e-5)
            cv2.imshow(nameWindow, color_img)
            frame_count += 1
            elapsed = time.time() - start_loop
            to_wait = max(1, int((interval - elapsed) * 1000))
            key = cv2.waitKey(to_wait) & 0xFF
            if key == 27 or key == ord('q'):
                break

    except KeyboardInterrupt:
        print("Interrupted by user.")

    finally:
        cv2.destroyAllWindows()
        print(f"Log saved to {logger.handlers[0].baseFilename}")
        print("Application closed cleanly.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--Path', type=str, default='.', help='Path.')
    parser.add_argument('--Interval', type=float, default=0.3, help='Interval.')
    parser.add_argument('--Height', type=int, default=64, help='Height.')
    parser.add_argument('--Num_samples', type=int, default=60, help='Num_samples.')
    parser.add_argument('--Mode', type=int, default=2, help='Mode.')
    args = parser.parse_args()
    
    
    if args.Mode==1:
        run_level1(args.Path,args.Interval, args.Num_samples)
    
    if args.Mode==2:
        run_level2(args.Path,args.Interval,args.Height)
