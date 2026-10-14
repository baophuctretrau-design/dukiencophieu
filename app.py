import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

# Cố định seed để ổn định kết quả
seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)
random.seed(seed)

st.title("Ứng dụng Dự báo Giá Cổ Phiếu 10 Ngày Tới")

# Cấu hình tham số từ người dùng
ticker = st.text_input("Mã cổ phiếu (VD: VNM.VN)", "VNM.VN")
start_date = st.date_input("Ngày bắt đầu", pd.to_datetime("2018-01-01"))
end_date = st.date_input("Ngày kết thúc (dữ liệu hiện tại)", pd.to_datetime("2026-10-14"))
window_size = st.slider("Cửa sổ thời gian (Window Size)", 30, 90, 60)
epochs_input = st.slider("Số lượng Epochs", 5, 50, 10)

if st.button("Bắt đầu Dự báo 10 ngày tới"):
    with st.spinner("Đang tải dữ liệu và huấn luyện mô hình..."):
        # Tải dữ liệu
        df = yf.download(ticker, start=start_date, end=end_date)
        if df.empty:
            st.error("Không tìm thấy dữ liệu cho mã cổ phiếu này!")
        else:
            data = df[['Close']].values
            
            # Chuẩn hóa toàn bộ dữ liệu[cite: 1]
            scaler = MinMaxScaler()
            data_scaled = scaler.fit_transform(data)

            # Hàm tạo chuỗi dữ liệu[cite: 1]
            def create_sequences(data, window_size):
                X, y = [], []
                for i in range(window_size, len(data)):
                    X.append(data[i-window_size:i])
                    y.append(data[i])
                return np.array(X), np.array(y)

            X_train, y_train = create_sequences(data_scaled, window_size)
            X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

            # Chia tập validation[cite: 1]
            val_size = int(0.1 * len(X_train))
            X_val = X_train[-val_size:]
            y_val = y_train[-val_size:]
            X_train_final = X_train[:-val_size]
            y_train_final = y_train[:-val_size]

            # Xây dựng mô hình LSTM[cite: 1]
            model = Sequential()
            model.add(LSTM(50, return_sequences=True, input_shape=(window_size, 1)))
            model.add(LSTM(50))
            model.add(Dense(1))
            model.compile(optimizer='adam', loss='mean_squared_error')

            early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

            # Huấn luyện mô hình[cite: 1]
            model.fit(
                X_train_final, y_train_final,
                validation_data=(X_val, y_val),
                epochs=epochs_input,
                batch_size=32,
                shuffle=False,
                callbacks=[early_stop],
                verbose=0
            )

            # Lấy 60 ngày cuối cùng để dự báo tương lai[cite: 1]
            last_60_days = data_scaled[-window_size:]
            last_60_days = last_60_days.reshape(1, window_size, 1)
            future_predictions = []

            # Vòng lặp dự báo 10 ngày tới[cite: 1]
            for i in range(10):
                next_pred = model.predict(last_60_days, verbose=0)
                future_predictions.append(next_pred[0,0])
                # Cập nhật lại cửa sổ thời gian với giá trị vừa dự báo[cite: 1]
                last_60_days = np.append(last_60_days[:,1:,:], next_pred.reshape(1,1,1), axis=1)

            future_predictions = np.array(future_predictions).reshape(-1,1)
            future_predictions = scaler.inverse_transform(future_predictions)

            st.success("Huấn luyện và dự báo thành công!")

            # Hiển thị kết quả dạng bảng
            st.subheader("Bảng Dự báo Giá Cổ Phiếu (10 ngày tiếp theo)")
            df_future = pd.DataFrame(
                future_predictions, 
                columns=["Giá dự báo"], 
                index=[f"Ngày T+{i+1}" for i in range(10)]
            )
            st.dataframe(df_future)

            # Vẽ biểu đồ 60 ngày qua và 10 ngày tới
            st.subheader("Biểu đồ Dự báo 10 ngày tới")
            real_data_unscaled = data[-window_size:]
            
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(range(1, window_size + 1), real_data_unscaled, label=f"Giá thật ({window_size} ngày qua)", color='blue')
            future_x = range(window_size + 1, window_size + 11)
            ax.plot(future_x, future_predictions, marker='o', label="Dự báo 10 ngày tới", color='red')
            
            ax.set_title(f"Dự báo giá cổ phiếu {ticker} trong 10 ngày tới")
            ax.set_xlabel("Thời gian (Ngày)")
            ax.set_ylabel("Giá cổ phiếu")
            ax.legend()
            st.pyplot(fig)
