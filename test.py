from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.cluster import KMeans
import data
# 示例数据
# 注意：这里的数据仅供示例，实际应用中需要根据你的数据进行替换
from sklearn.datasets import make_blobs
X, y = make_blobs(n_samples=300, centers=3, random_state=42)

# 使用K均值聚类作为示例聚类算法
kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(X)

# 计算轮廓系数
silhouette_avg = silhouette_score(X, labels)
print(f"Silhouette Coefficient: {silhouette_avg}")

# 计算Calinski-Harabasz指数
calinski_harabasz_score_val = calinski_harabasz_score(X, labels)
print(f"Calinski-Harabasz Index: {calinski_harabasz_score_val}")

# 计算Davies-Bouldin指数
davies_bouldin_score_val = davies_bouldin_score(X, labels)
print(f"Davies-Bouldin Index: {davies_bouldin_score_val}")
