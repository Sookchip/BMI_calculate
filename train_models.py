import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.preprocessing import StandardScaler

# Đọc dữ liệu từ tệp CSV
dataset = pd.read_csv('data/data_BMI.csv')

# Giả sử các cột là 'Height', 'Weight' và 'BMI'
X = dataset[['Height', 'Weight']]
y = dataset['BMI']

# Chia dữ liệu thành tập huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Chuẩn hóa dữ liệu cho Neural Network
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Huấn luyện các mô hình cơ bản
linear_model = LinearRegression()
ridge_model = Ridge(alpha=1.0)
mlp_model = MLPRegressor(hidden_layer_sizes=(100,), max_iter=2000, random_state=42)

linear_model.fit(X_train, y_train)
ridge_model.fit(X_train, y_train)
mlp_model.fit(X_train_scaled, y_train)

# Tạo mô hình Stacking
estimators = [
    ('linear', linear_model),
    ('ridge', ridge_model),
    ('mlp', mlp_model)
]
stacking_model = StackingRegressor(estimators=estimators, final_estimator=LinearRegression())

# Huấn luyện mô hình Stacking
stacking_model.fit(X_train_scaled, y_train)

# Lưu tất cả các mô hình và scaler
joblib.dump(linear_model, 'models/linear_model.pkl')
joblib.dump(ridge_model, 'models/ridge_model.pkl')
joblib.dump(mlp_model, 'models/mlp_model.pkl')
joblib.dump(stacking_model, 'models/stacking_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
