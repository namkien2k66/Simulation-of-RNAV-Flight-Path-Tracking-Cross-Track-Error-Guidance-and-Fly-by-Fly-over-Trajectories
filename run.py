import os
from RNAP1.experiments import (
    exp1_capture,
    exp2_flyby_vs_flyover,
    exp3_guidance_comparison,
    exp4_parameter_sweeps,
    exp5_wind
)

# Tạo thư mục 'out' để lưu kết quả
out_dir = "out"
os.makedirs(out_dir, exist_ok=True)

print("=== BẮT ĐẦU CHẠY MÔ PHỎNG RNAV ===")

print("[1/5] Đang chạy Thí nghiệm 1: Bắt đường bay...")
exp1_capture(out_dir)

print("[2/5] Đang chạy Thí nghiệm 2: So sánh Fly-by vs Fly-over...")
exp2_flyby_vs_flyover(out_dir)

print("[3/5] Đang chạy Thí nghiệm 3: So sánh luật điều hướng...")
exp3_guidance_comparison(out_dir)

print("[4/5] Đang chạy Thí nghiệm 4: Quét tham số...")
exp4_parameter_sweeps(out_dir)

print("[5/5] Đang chạy Thí nghiệm 5: Ảnh hưởng của gió...")
exp5_wind(out_dir)

print(f"=== HOÀN THÀNH! Kết quả được lưu tại thư mục: {out_dir} ===")