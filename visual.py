import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, NMF
from sklearn.manifold import MDS, Isomap, LocallyLinearEmbedding, TSNE
from sklearn.preprocessing import StandardScaler
import time
import matplotlib
import pandas as pd
import requests
import io
import zipfile
import os

# 设置matplotlib支持中文显示
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']  # 优先使用的字体列表
matplotlib.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 设置随机种子保证结果可复现
np.random.seed(42)

# 加载MovieLens数据集
def load_movielens_data():
    print("加载MovieLens数据集...")
    
    # 如果数据已经下载过，就直接读取本地文件
    if os.path.exists("ml-latest-small"):
        ratings_df = pd.read_csv("ml-latest-small/ratings.csv")
        movies_df = pd.read_csv("ml-latest-small/movies.csv")
    else:
        # 否则从网络下载
        url = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
        print(f"从 {url} 下载数据...")
        response = requests.get(url)
        with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
            zip_ref.extractall(".")
        ratings_df = pd.read_csv("ml-latest-small/ratings.csv")
        movies_df = pd.read_csv("ml-latest-small/movies.csv")
    
    print(f"评分数据: {ratings_df.shape[0]} 评分, {ratings_df['userId'].nunique()} 用户, {ratings_df['movieId'].nunique()} 电影")
    
    # 只选择评分数量较多的电影和活跃用户
    min_user_ratings = 50  # 用户至少评价过的电影数
    min_movie_ratings = 100  # 电影至少被评价的次数
    
    # 过滤电影和用户
    movie_counts = ratings_df['movieId'].value_counts()
    user_counts = ratings_df['userId'].value_counts()
    
    valid_movies = movie_counts[movie_counts >= min_movie_ratings].index
    valid_users = user_counts[user_counts >= min_user_ratings].index
    
    filtered_ratings = ratings_df[
        ratings_df['movieId'].isin(valid_movies) & 
        ratings_df['userId'].isin(valid_users)
    ]
    
    # 创建用户-电影评分矩阵
    rating_matrix = filtered_ratings.pivot(
        index='userId', columns='movieId', values='rating'
    ).fillna(0)  # 填充未评分的项为0
    
    print(f"过滤后的评分矩阵形状: {rating_matrix.shape}")
    
    # 获取对应的电影信息
    filtered_movies = movies_df[movies_df['movieId'].isin(rating_matrix.columns)]
    
    # 提取类型作为标签
    # 每部电影可能有多个类型，我们只取第一个类型作为标签
    movie_genres = filtered_movies['genres'].apply(lambda x: x.split('|')[0])
    unique_genres = movie_genres.unique()
    genre_to_id = {genre: i for i, genre in enumerate(unique_genres)}
    genre_ids = movie_genres.map(genre_to_id).values
    
    # 返回评分矩阵和电影类型
    return rating_matrix.values, genre_ids, list(unique_genres)

# 可视化函数
def visualize(X_embedded, labels, label_names, title, file_name=None):
    plt.figure(figsize=(12, 10))
    
    # 使用不同颜色表示不同电影类型
    scatter = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=labels, cmap='tab20', 
               alpha=0.8, s=50)
    
    # 添加图例
    handles, _ = scatter.legend_elements()
    legend_labels = [label_names[i] for i in range(len(label_names))]
    plt.legend(handles, legend_labels, loc="upper right", title="电影类型")
    
    plt.title(title)
    plt.tight_layout()
    if file_name:
        plt.savefig(file_name)
    plt.show()

# 1. 主成分分析 (PCA)
def apply_pca(X):
    print("\n应用主成分分析(PCA)...")
    start_time = time.time()
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    print(f"PCA 完成，耗时: {time.time() - start_time:.2f} 秒")
    print(f"解释方差占比: {np.sum(pca.explained_variance_ratio_):.4f}")
    return X_pca

# 2. 多维尺度分析 (MDS)
def apply_mds(X):
    print("\n应用多维尺度分析(MDS)...")
    start_time = time.time()
    mds = MDS(n_components=2, n_jobs=-1, random_state=42)
    X_mds = mds.fit_transform(X)
    print(f"MDS 完成，耗时: {time.time() - start_time:.2f} 秒")
    return X_mds

# 3. 非负矩阵分解 (NMF)
def apply_nmf(X):
    print("\n应用非负矩阵分解(NMF)...")
    # 确保数据非负 (对于评分数据通常已经是非负的)
    X_pos = np.maximum(X, 0)
    start_time = time.time()
    nmf = NMF(n_components=2, init='random', random_state=42)
    X_nmf = nmf.fit_transform(X_pos)
    print(f"NMF 完成，耗时: {time.time() - start_time:.2f} 秒")
    return X_nmf

# 4. 等度量映射 (Isomap)
def apply_isomap(X):
    print("\n应用等度量映射(Isomap)...")
    start_time = time.time()
    isomap = Isomap(n_components=2, n_neighbors=10, n_jobs=-1)
    X_isomap = isomap.fit_transform(X)
    print(f"Isomap 完成，耗时: {time.time() - start_time:.2f} 秒")
    return X_isomap

# 5. 局部线性嵌入 (LLE)
def apply_lle(X):
    print("\n应用局部线性嵌入(LLE)...")
    start_time = time.time()
    lle = LocallyLinearEmbedding(n_components=2, n_neighbors=10, n_jobs=-1, random_state=42)
    X_lle = lle.fit_transform(X)
    print(f"LLE 完成，耗时: {time.time() - start_time:.2f} 秒")
    return X_lle

# 6. t-SNE
def apply_tsne(X):
    print("\n应用t-SNE...")
    start_time = time.time()
    tsne = TSNE(n_components=2, random_state=42, n_jobs=-1)
    X_tsne = tsne.fit_transform(X)
    print(f"t-SNE 完成，耗时: {time.time() - start_time:.2f} 秒")
    return X_tsne

def main():
    # 加载MovieLens数据
    X, genre_ids, genre_names = load_movielens_data()
    
    # 预处理数据 - 转置矩阵，分析电影相似性而不是用户相似性
    # 转置后每行代表一部电影，每列代表一个用户对该电影的评分
    X = X.T
    
    print(f"数据矩阵形状: {X.shape}, 表示 {X.shape[0]} 部电影 × {X.shape[1]} 个用户")
    print(f"电影类型数量: {len(genre_names)}, 类型: {', '.join(genre_names)}")
    
    # 应用所有降维方法
    
    # 1. PCA
    X_pca = apply_pca(X)
    visualize(X_pca, genre_ids, genre_names, "主成分分析 (PCA) - 电影分布", "pca_movies.png")
    
    # 2. MDS - 可能会很慢，根据矩阵大小可能需要跳过
    try:
        X_mds = apply_mds(X)
        visualize(X_mds, genre_ids, genre_names, "多维尺度分析 (MDS) - 电影分布", "mds_movies.png")
    except Exception as e:
        print(f"MDS 出错: {e}，矩阵可能太大，跳过此方法")
    
    # 3. NMF - 特别适合推荐系统数据
    X_nmf = apply_nmf(X)
    visualize(X_nmf, genre_ids, genre_names, "非负矩阵分解 (NMF) - 电影分布", "nmf_movies.png")
    
    # 4. Isomap
    try:
        X_isomap = apply_isomap(X)
        visualize(X_isomap, genre_ids, genre_names, "等度量映射 (Isomap) - 电影分布", "isomap_movies.png")
    except Exception as e:
        print(f"Isomap 出错: {e}，矩阵可能太大，跳过此方法")
    
    # 5. LLE
    try:
        X_lle = apply_lle(X)
        visualize(X_lle, genre_ids, genre_names, "局部线性嵌入 (LLE) - 电影分布", "lle_movies.png")
    except Exception as e:
        print(f"LLE 出错: {e}，矩阵可能太大，跳过此方法")
    
    # 6. t-SNE
    X_tsne = apply_tsne(X)
    visualize(X_tsne, genre_ids, genre_names, "t-SNE - 电影分布", "tsne_movies.png")
    
    print("\n所有降维方法已完成！")

if __name__ == "__main__":
    main()

