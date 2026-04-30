# model_creator.py - Huấn luyện Isolation Forest và Lưu trữ

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest  # Thư viện Isolation Forest
import sys

# --- 1. TẠO DỮ LIỆU GIẢ LẬP ---
# (Sử dụng 10 tính năng như ví dụ ban đầu)
def create_sample_data(n_samples=1000, n_features=10):
    np.random.seed(42)
    normal_data = np.random.randn(n_samples, n_features) * 2 + 5 
    
    # Dữ liệu bất thường: Rất xa lạ
    anomaly_data = np.random.uniform(low=20, high=30, size=(int(n_samples * 0.1), n_features))
    
    data = np.vstack([normal_data, anomaly_data])
    labels = np.array([0] * n_samples + [1] * anomaly_data.shape[0])
    
    df = pd.DataFrame(data, columns=[f'feature_{i+1}' for i in range(n_features)])
    df['label'] = labels
    print(f"Dữ liệu mẫu đã tạo: {len(df)} hàng, {n_features} tính năng.")
    return df

# --- 2. CẤU TRÚC VÀ HUẤN LUYỆN ISOLATION FOREST ---

def build_and_train_isolation_forest(data):
    # CHỈ LẤY DỮ LIỆU BÌNH THƯỜNG ĐỂ HUẤN LUYỆN
    normal_data = data[data['label'] == 0].drop(columns=['label'])
    
    # Chuẩn hóa dữ liệu
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(normal_data)
    
    input_dim = scaled_data.shape[1]

    # 1. Khởi tạo mô hình Isolation Forest
    # contamination=0.05 giả định 5% mẫu huấn luyện là bất thường (để tính ngưỡng)
    model = IsolationForest(contamination=0.05, random_state=42)
    
    # 2. Huấn luyện mô hình
    print("\n--- Bắt đầu Huấn luyện Isolation Forest ---")
    model.fit(scaled_data)

    # 3. Tính toán Ngưỡng
    # decision_function: Trả về điểm bất thường. Điểm càng ÂM, càng bất thường.
    anomaly_scores = model.decision_function(scaled_data)
    
    # Ngưỡng: Đặt tại 5th percentile của điểm bất thường
    threshold = np.percentile(anomaly_scores, 5) 

    return model, scaler, threshold, input_dim

# --- 3. CHẠY VÀ LƯU TRỮ ---

if __name__ == '__main__':
    # Đảm bảo cài đặt thư viện cần thiết
    if 'sklearn.ensemble' not in sys.modules:
        print("Lỗi: Vui lòng cài đặt các thư viện cần thiết: pip install pandas numpy scikit-learn joblib")
        sys.exit(1)
        
    try:
        sample_df = create_sample_data(n_features=10) 
        model, scaler, threshold, input_dim = build_and_train_isolation_forest(sample_df)
    except Exception as e:
        print(f"Xảy ra lỗi trong quá trình huấn luyện: {e}")
        sys.exit(1)

    # 4. Lưu trữ các thành phần
    try:
        joblib.dump(model, 'isolation_forest_model.pkl') # Lưu mô hình IF
        joblib.dump(scaler, 'scaler.pkl') 
        with open('threshold.txt', 'w') as f:
            f.write(str(threshold))
        
        print("\n--- HOÀN TẤT LƯU TRỮ ---")
        print(f"Mô hình được cấu hình với {input_dim} tính năng.")
        print(f"Đã lưu: isolation_forest_model.pkl, scaler.pkl, threshold.txt")
        print(f"Ngưỡng lỗi phát hiện: {threshold:.4f} (Điểm dưới ngưỡng này là BẤT THƯỜNG)")
        print("Bây giờ bạn có thể chạy 'python detector.py' để kiểm tra.")
        
    except Exception as e:
        print(f"Lỗi khi lưu file: {e}")