# detector.py - Công cụ Phát hiện IP Lạ với Trực quan hóa

import joblib
import numpy as np
import sys
import os
from rich.console import Console
from rich.table import Table
from rich.text import Text

# --- Cài đặt Console (để in ra Terminal đẹp hơn) ---
console = Console()

# --- 1. TẢI CÁC TÀI NGUYÊN (Giữ nguyên) ---
MODEL_PATH = 'isolation_forest_model.pkl'
SCALER_PATH = 'scaler.pkl'
THRESHOLD_PATH = 'threshold.txt'

try:
    loaded_model = joblib.load(MODEL_PATH)
    loaded_scaler = joblib.load(SCALER_PATH)
    with open(THRESHOLD_PATH, 'r') as f:
        loaded_threshold = float(f.read())
except FileNotFoundError as e:
    console.print(f"[bold red]Lỗi:[/bold red] Không tìm thấy tệp tài nguyên cần thiết ({e.filename}).")
    console.print("Vui lòng chạy 'python model_creator.py' trước.")
    sys.exit(1)
except Exception as e:
    console.print(f"[bold red]Lỗi khi tải tài nguyên:[/bold red] {e}")
    sys.exit(1)

input_dim = loaded_scaler.n_features_in_
results_history = [] # Danh sách để lưu trữ kết quả của nhiều IP

# --- 2. HÀM PHÁT HIỆN VÀ GHI NHẬN ---

def process_single_ip(ip_name, data_values, model, scaler, threshold):
    # Chuyển đổi chuỗi đầu vào THÔ thành mảng numpy
    try:
        data_array = np.array([float(x.strip()) for x in data_values.split(',')])
        
        if data_array.size != input_dim:
             console.print(f"[bold yellow]⚠️ Lỗi:[/bold yellow] Số lượng tính năng không khớp. Cần {input_dim} giá trị.")
             return None
             
    except ValueError:
        console.print("[bold yellow]⚠️ Lỗi:[/bold yellow] Đầu vào không hợp lệ.")
        return None

    # Chuẩn hóa dữ liệu thô
    data_point = data_array.reshape(1, -1)
    scaled_data = scaler.transform(data_point)

    # Tính Điểm Bất thường IF (Score)
    anomaly_score = model.decision_function(scaled_data)[0]

    # Quyết định
    is_anomaly = anomaly_score < threshold
    status_text = "BẤT THƯỜNG (LẠ) 🚨" if is_anomaly else "BÌNH THƯỜNG ✅"
    color = "bold red" if is_anomaly else "bold green"
    
    # Ghi nhận kết quả
    result = {
        "IP": ip_name,
        "Score": anomaly_score,
        "Status": status_text,
        "Color": color
    }
    results_history.append(result)
    
    console.print(f"[{color}]{ip_name}:[/] Điểm Score: [bold]{anomaly_score:.4f}[/], Trạng thái: [{color}]{status_text}[/]")
    return result

# --- 3. HÀM TRỰC QUAN HÓA (BIỂU ĐỒ) ---

def display_results_visualization(results, threshold):
    if not results:
        console.print("[yellow]Chưa có dữ liệu để trực quan hóa.[/yellow]")
        return
    
    # Tạo bảng so sánh
    table = Table(title="\nBIỂU ĐỒ TRỰC QUAN HÓA ĐỘ BẤT THƯỜNG", show_lines=True)
    table.add_column("IP / ID", style="cyan", justify="left")
    table.add_column("Điểm Bất thường (Score)", style="magenta")
    table.add_column("Trạng thái", style="bold")
    table.add_column("Biểu đồ Thanh", style="yellow")
    
    # Tìm min/max score để chuẩn hóa cho biểu đồ thanh
    scores = [r['Score'] for r in results]
    min_score = min(scores)
    max_score = max(scores)
    
    # Khảo sát phạm vi scores để vẽ biểu đồ thanh có ý nghĩa hơn
    score_range = max_score - min_score
    if score_range == 0:
        score_range = 1 # Tránh chia cho 0

    for r in sorted(results, key=lambda x: x['Score']): # Sắp xếp để dễ so sánh
        score_normalized = (r['Score'] - min_score) / score_range
        
        # Vẽ biểu đồ thanh dựa trên độ lớn của điểm số (tối đa 20 ký tự #)
        bar_length = int(score_normalized * 20)
        bar = Text("█" * bar_length, style=r['Color'])
        bar.append(f" ({score_normalized:.2f})")
        
        table.add_row(
            r['IP'],
            f"{r['Score']:.4f}",
            Text(r['Status'], style=r['Color']),
            bar
        )
    
    console.print(table)
    console.print(f"[bold yellow]Ngưỡng:[/bold yellow] [bold magenta]{threshold:.4f}[/] (Điểm dưới ngưỡng là bất thường)")

# --- 4. CHẠY CLI TƯƠNG TÁC ---

if __name__ == '__main__':
    console.print("=" * 50, style="bold blue")
    console.print("--- Công cụ Phát hiện IP Lạ (Isolation Forest CLI) ---", style="bold yellow")
    console.print(f"Mô hình đang sử dụng [bold]{input_dim}[/] tính năng. Ngưỡng: [bold magenta]{loaded_threshold:.4f}[/]")
    console.print("=" * 50, style="bold blue")
    
    ip_counter = 1
    
    while True:
        try:
            user_input = console.input(f"Nhập [bold cyan]ID/Tên IP[/] và [bold]{input_dim}[/] giá trị THÔ (cách nhau bằng dấu phẩy) hoặc 'show'/'exit':\n> ")
            
            if user_input.lower() == 'exit':
                console.print("\n[bold red]Đã thoát công cụ.[/bold red]")
                break
            
            if user_input.lower() == 'show':
                display_results_visualization(results_history, loaded_threshold)
                continue
            
            # Phân tách đầu vào: Giả sử người dùng nhập: Tên_IP, giá_trị_1, giá_trị_2, ...
            parts = [p.strip() for p in user_input.split(',')]
            
            if len(parts) == input_dim + 1:
                ip_name = parts[0]
                data_string = ",".join(parts[1:])
                process_single_ip(ip_name, data_string, loaded_model, loaded_scaler, loaded_threshold)
                ip_counter += 1
            elif len(parts) == input_dim:
                # Nếu người dùng chỉ nhập giá trị, sử dụng ID tự động
                data_string = user_input
                ip_name = f"IP_Test_{ip_counter}"
                process_single_ip(ip_name, data_string, loaded_model, loaded_scaler, loaded_threshold)
                ip_counter += 1
            else:
                console.print(f"[bold yellow]⚠️ Lỗi:[/bold yellow] Định dạng nhập sai. Vui lòng nhập: [cyan]Tên_IP, [yellow]10_giá_trị_THÔ[/] (hoặc chỉ 10 giá trị).")
                
        except KeyboardInterrupt:
            console.print("\n[bold red]Đã thoát công cụ.[/bold red]")
            break