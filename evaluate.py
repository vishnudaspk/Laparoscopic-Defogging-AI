import os
import time
import cv2
import numpy as np
import torch
import pandas as pd
from skimage.metrics import structural_similarity as ssim

from main import desmoker

def evaluate():
    # Setup paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "scripts", "checkpoints", "pix2pix_laparoscopy_dc", "best_net_G.pth")
    input_dir = os.path.join(base_dir, "datasets", "laparoscopy", "input")
    gt_dir = os.path.join(base_dir, "datasets", "laparoscopy", "output")
    eval_dir = os.path.join(base_dir, "Evaluation")
    
    os.makedirs(eval_dir, exist_ok=True)
    
    # Initialize model
    print("Loading model...")
    try:
        model = desmoker(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
        
    results = []
    
    # Get list of images
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f"Found {len(image_files)} images for evaluation.")
    
    for filename in image_files:
        in_path = os.path.join(input_dir, filename)
        gt_path = os.path.join(gt_dir, filename)
        
        if not os.path.exists(gt_path):
            print(f"Ground truth missing for {filename}, skipping.")
            continue
            
        img_bgr = cv2.imread(in_path)
        gt_bgr = cv2.imread(gt_path)
        
        if img_bgr is None or gt_bgr is None:
            continue
            
        h, w = img_bgr.shape[:2]
        
        # Prepare for model
        img_256 = cv2.resize(img_bgr, (256, 256))
        gt_256 = cv2.resize(gt_bgr, (256, 256))
        
        # Inference
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            
        t0 = time.perf_counter()
        # Default params
        result_rgb, _ = model.apply(img_256)
        inference_time = time.perf_counter() - t0
        
        # GPU Memory
        gpu_memory_mb = 0
        if torch.cuda.is_available():
            gpu_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
            
        # Convert RGB back to BGR for metric calculation
        result_bgr = cv2.cvtColor(result_rgb[:, :, :3], cv2.COLOR_RGB2BGR)
        
        # Calculate PSNR
        psnr_val = cv2.PSNR(gt_256, result_bgr)
        
        # Calculate SSIM (convert to gray for SSIM, or calculate multichannel)
        # multichannel=True is deprecated in newer skimage, use channel_axis=-1
        try:
            ssim_val = ssim(gt_256, result_bgr, channel_axis=-1, data_range=255)
        except TypeError:
            ssim_val = ssim(gt_256, result_bgr, multichannel=True, data_range=255)
            
        fps = 1.0 / inference_time if inference_time > 0 else 0
        
        results.append({
            "Filename": filename,
            "PSNR (dB)": psnr_val,
            "SSIM": ssim_val,
            "Inference Time (s)": inference_time,
            "FPS": fps,
            "GPU Memory (MB)": gpu_memory_mb
        })
        print(f"Processed {filename} | PSNR: {psnr_val:.2f} | SSIM: {ssim_val:.4f} | FPS: {fps:.1f}")

    if not results:
        print("No valid image pairs found.")
        return
        
    df = pd.DataFrame(results)
    
    # Calculate averages
    avg_psnr = df["PSNR (dB)"].mean()
    avg_ssim = df["SSIM"].mean()
    avg_inf_time = df["Inference Time (s)"].mean()
    avg_fps = df["FPS"].mean()
    avg_gpu_mem = df["GPU Memory (MB)"].mean()
    
    # Save CSV
    csv_path = os.path.join(eval_dir, "results.csv")
    df.to_csv(csv_path, index=False)
    
    # Save Markdown Report
    report_path = os.path.join(eval_dir, "report.md")
    with open(report_path, "w") as f:
        f.write("# Laparoscopy Defogging AI - Evaluation Report\n\n")
        f.write(f"**Total Images Evaluated**: {len(df)}\n\n")
        f.write("## Aggregate Metrics\n\n")
        f.write("| Metric | Average Value |\n")
        f.write("|---|---|\n")
        f.write(f"| **PSNR (Reconstruction Quality)** | {avg_psnr:.2f} dB |\n")
        f.write(f"| **SSIM (Structural Preservation)** | {avg_ssim:.4f} |\n")
        f.write(f"| **Inference Time (Speed)** | {avg_inf_time:.4f} s |\n")
        f.write(f"| **FPS (Real-time Capability)** | {avg_fps:.1f} |\n")
        f.write(f"| **Peak GPU Memory** | {avg_gpu_mem:.1f} MB |\n\n")
        f.write("## Detailed Results\n\n")
        f.write("Detailed per-image results can be found in `results.csv`.\n")

    print(f"\nEvaluation complete. Results saved to {eval_dir}")
    print(f"Average PSNR: {avg_psnr:.2f} | Average SSIM: {avg_ssim:.4f}")

if __name__ == "__main__":
    evaluate()
