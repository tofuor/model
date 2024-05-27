from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

# 生成一個二分類數據集
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=42)

# 分割數據集為訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 標準化特徵
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 初始化 SVC 模型，設置 C 參數
model = SVC(C=1.0, kernel='linear')

# 訓練模型
model.fit(X_train, y_train)

# 測試模型
accuracy = model.score(X_test, y_test)
print(f"Model accuracy: {accuracy*100:.2f}%")
