import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun', 'KaiTi']
plt.rcParams['axes.unicode_minus'] = False

# 加载数据
data = np.load('preprocess\embedding\output\sampled_output_embeddings.npz')

# 访问不同类型的数据
# indices = data['indices']
# sequences = data['sequences']
diff_counts = data['diff_counts']
mean_log10Ka_values = data['mean_log10Ka']
# embeddings = data['embeddings']  # 这是一个形状为 [n, 100] 的数组
# print(embeddings.shape)  # 输出形状，应该是 (n, 100)

mask = mean_log10Ka_values < 18
diff_counts_selected = diff_counts[mask]

# 绘制 diff_counts 分布直方图
plt.figure(figsize=(8, 5))
plt.hist(diff_counts_selected, bins=30, color='orange', edgecolor='black')
plt.xlabel('diff_counts')
plt.ylabel('样本数')
plt.title('diff_counts 分布')
plt.grid(True)
plt.tight_layout()
plt.show()