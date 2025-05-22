import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Union, Any, Optional
import os
import time
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
import gc

class ProteinEncoder:
    def __init__(self, protvec_file_path: str):
        """
        初始化蛋白质编码器
        
        Args:
            protvec_file_path: ProtVec向量文件路径(TSV格式)
        """
        # 加载ProtVec向量
        self.trigram_vectors = self._load_protvec_optimized(protvec_file_path)
        
        if self.trigram_vectors:
            # 获取向量维度
            self.vector_dim = self.vectors_array.shape[1]
            print(f"已加载ProtVec向量，维度: {self.vector_dim}")
    
    def _load_protvec_optimized(self, file_path: str) -> Dict[str, np.ndarray]:
        """
        优化的ProtVec向量文件加载函数，使用float32数据类型并将字典转换为数组
        
        Args:
            file_path: ProtVec文件路径
        
        Returns:
            字典，键为三联体字符串，值为对应的向量表示
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"找不到ProtVec文件: {file_path}")
        
        try:
            print(f"正在加载ProtVec向量文件: {file_path}")
            # 使用chunksize参数分批读取大文件
            trigram_vectors = {}
            
            # 首先读取文件以获取向量维度
            with open(file_path, 'r') as f:
                header = f.readline().strip()
                first_line = f.readline().strip()
                first_parts = first_line.split('\t')
                vector_dim = len(first_parts) - 1
            
            # 预分配足够的内存来存储所有向量
            all_trigrams = []
            all_vectors = []
            
            for chunk in pd.read_csv(file_path, sep='\t', index_col=0, chunksize=10000):
                # 使用float32而非默认的float64来减少内存使用
                chunk_vectors = chunk.values.astype(np.float32)
                all_trigrams.extend(chunk.index.tolist())
                all_vectors.append(chunk_vectors)
            
            # 合并所有向量
            vectors_array = np.vstack(all_vectors)
            
            # 创建三联体字典，使用优化的数据结构
            trigram_vectors = {trigram: idx for idx, trigram in enumerate(all_trigrams)}
            
            # 保存向量数组
            self.vectors_array = vectors_array
            
            # 创建反向映射，用于快速查找
            self.trigram_to_index = trigram_vectors
            
            print(f"已成功加载{len(trigram_vectors)}个三联体向量")
            return trigram_vectors
            
        except Exception as e:
            print(f"加载ProtVec文件时出错: {e}")
            return {}
    
    def sequence_to_trigrams(self, sequence: str) -> List[str]:
        """将蛋白质序列转换为三联体列表"""
        return [sequence[i:i+3] for i in range(len(sequence)-2)]
    
    def get_trigram_embeddings(self, sequence: str) -> np.ndarray:
        """
        获取序列的三联体嵌入表示，优化版本
        
        Args:
            sequence: 蛋白质氨基酸序列
            
        Returns:
            三联体嵌入矩阵
        """
        trigrams = self.sequence_to_trigrams(sequence)
        
        # 有效索引列表
        valid_indices = []
        for trigram in trigrams:
            idx = self.trigram_to_index.get(trigram, -1)
            if idx != -1:
                valid_indices.append(idx)
        
        # 如果没有有效的三联体，返回零向量
        if not valid_indices:
            return np.zeros((0, self.vector_dim), dtype=np.float32)
        
        # 使用索引直接从数组获取嵌入
        embeddings = self.vectors_array[valid_indices]
        
        return embeddings
    
    def get_pentamer_embeddings_optimized(self, sequence: str, method: str = "sum") -> np.ndarray:
        """
        获取序列的五联体嵌入表示，通过组合三个连续的三联体，优化版本
        
        Args:
            sequence: 蛋白质氨基酸序列
            method: 组合方法，"sum"表示求和，"mean"表示求平均值
            
        Returns:
            五联体嵌入矩阵，形状为 [序列长度-4, vector_dim]
        """
        if len(sequence) < 5:
            return np.zeros((0, self.vector_dim), dtype=np.float32)
        
        # 获取三联体嵌入
        trigram_embeddings = self.get_trigram_embeddings(sequence)
        
        if len(trigram_embeddings) < 3:
            return np.zeros((0, self.vector_dim), dtype=np.float32)
        
        # 计算五联体嵌入的数量
        num_pentamers = len(sequence) - 4
        
        # 预分配数组
        pentamer_embeddings = np.zeros((num_pentamers, self.vector_dim), dtype=np.float32)
        
        # 高效计算五联体嵌入
        for i in range(min(num_pentamers, len(trigram_embeddings)-2)):
            trigram_group = trigram_embeddings[i:i+3]
            if method == "sum":
                pentamer_embeddings[i] = np.sum(trigram_group, axis=0)
            else:  # "mean"
                pentamer_embeddings[i] = np.mean(trigram_group, axis=0)
        
        return pentamer_embeddings

    def _process_sequence_efficiently(self, 
                                    row_tuple: Tuple[int, pd.Series], 
                                    use_pentamers: bool, 
                                    combine_method: str) -> Tuple[int, np.ndarray]:
        """
        处理单个序列的辅助函数，保留所有五联体嵌入
        
        Args:
            row_tuple: 行索引和序列数据的元组
            use_pentamers: 是否使用五联体表示
            combine_method: 五联体组合方法
            
        Returns:
            行索引和嵌入表示的元组，嵌入是二维数组 [n_pentamers, vector_dim]
        """
        idx, row = row_tuple
        sequence = row['aim_seq']
        
        try:
            if use_pentamers:
                # 直接返回所有五联体嵌入
                pentamer_embeddings = self.get_pentamer_embeddings_optimized(sequence, method=combine_method)
                return idx, pentamer_embeddings
            else:
                # 三联体表示
                trigram_embeddings = self.get_trigram_embeddings(sequence)
                return idx, trigram_embeddings
                
        except Exception as e:
            print(f"处理序列时出错 (索引 {idx}): {e}")
            return idx, np.zeros((0, self.vector_dim), dtype=np.float32)

    def process_protein_dataset_optimized(self, 
                                        input_file: str, 
                                        output_file: str, 
                                        use_pentamers: bool = True,
                                        combine_method: str = "sum",
                                        n_processes: int = None,
                                        batch_size: int = 8000,
                                        chunksize: int = 50):
        """
        批量处理蛋白质序列数据集，保留所有五联体表示
        
        Args:
            input_file: 输入CSV文件路径
            output_file: 输出NPZ文件路径，无需扩展名
            use_pentamers: 是否使用五联体表示
            combine_method: 五联体组合方法，"sum"或"mean"
            n_processes: 进程数，默认使用CPU核心数-1
            batch_size: 每批处理的数据量
            chunksize: 多进程任务分配的块大小
        """
        try:
            # 设置进程数
            if n_processes is None:
                n_processes = max(1, mp.cpu_count() - 1)  # 默认使用CPU核心数-1
            
            print(f"将使用 {n_processes} 个进程进行并行处理")
            
            # 获取总行数以显示整体进度
            total_rows = sum(1 for _ in open(input_file)) - 1  # 减去标题行
            print(f"数据集共有 {total_rows} 条记录，将分批处理")
            
            # 创建输出文件目录（如果不存在）
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            # 为存储结果准备数据结构
            original_indices = []
            sequences = []
            diff_counts = []
            mean_log10Ka_values = []
            embeddings_list = []  # 存储每个序列的嵌入矩阵
            seq_lengths = []      # 存储每个序列的长度
            
            # 分批读取数据
            remaining_rows = total_rows
            batch_num = 1
            
            with mp.Pool(processes=n_processes) as pool:
                # 使用partial预设处理函数参数
                process_func = partial(
                    self._process_sequence_efficiently, 
                    use_pentamers=use_pentamers, 
                    combine_method=combine_method
                )
                
                # 按批次处理数据
                for chunk_df in pd.read_csv(input_file, chunksize=batch_size):
                    batch_start_time = time.time()
                    print(f"\n处理批次 {batch_num}, 大小: {len(chunk_df)} 行")
                    
                    # 准备批次数据
                    row_tuples = list(chunk_df.iterrows())
                    
                    # 使用imap_unordered提高性能并设置合适的chunksize
                    results = list(tqdm(
                        pool.imap_unordered(process_func, row_tuples, chunksize=chunksize),
                        total=len(row_tuples),
                        desc=f"批次 {batch_num}/{(total_rows + batch_size - 1) // batch_size}"
                    ))
                    
                    # 转换结果为字典
                    embedding_dict = {idx: embedding for idx, embedding in results}
                    
                    # 收集当前批次结果
                    for idx, row in chunk_df.iterrows():
                        if idx in embedding_dict:
                            sequence = row['aim_seq']
                            embedding_matrix = embedding_dict[idx]
                            
                            # 只保存有效的嵌入矩阵
                            if embedding_matrix.size > 0:
                                original_indices.append(row.get('index', idx))
                                sequences.append(sequence)
                                diff_counts.append(row['diff_count'])
                                mean_log10Ka_values.append(row['mean_log10Ka'])
                                embeddings_list.append(embedding_matrix)
                                seq_lengths.append(len(sequence))
                    
                    # 清理内存
                    del results, embedding_dict, row_tuples
                    gc.collect()
                    
                    # 更新进度信息
                    batch_end_time = time.time()
                    remaining_rows -= len(chunk_df)
                    batch_num += 1
                    
                    # 显示批次处理信息
                    print(f"批次处理完成，用时: {batch_end_time - batch_start_time:.2f}秒")
                    print(f"剩余约 {remaining_rows} 行待处理")
            
            # 转换为NumPy数组
            original_indices = np.array(original_indices, dtype=np.int64)
            sequences = np.array(sequences, dtype=object)
            diff_counts = np.array(diff_counts, dtype=np.int32)
            mean_log10Ka_values = np.array(mean_log10Ka_values, dtype=np.float32)
            seq_lengths = np.array(seq_lengths, dtype=np.int32)
            
            # 保存为NPZ文件
            np.savez_compressed(
                f"{output_file}.npz",
                indices=original_indices,
                sequences=sequences,
                diff_counts=diff_counts,
                mean_log10Ka=mean_log10Ka_values,
                seq_lengths=seq_lengths,
                embeddings=embeddings_list  # 直接存储嵌入矩阵列表
            )
            
            print(f"所有数据处理完成，结果已保存至: {output_file}.npz")
            
            # 显示保存的数据结构信息
            print("\n保存的数据结构:")
            print(f"处理的序列数量: {len(sequences)}")
            if len(embeddings_list) > 0:
                print(f"嵌入矩阵示例形状: {embeddings_list[0].shape} (第一条序列)")
                print(f"向量维度: {self.vector_dim}")
            print("数据字段: ['indices', 'sequences', 'diff_counts', 'mean_log10Ka', 'seq_lengths', 'embeddings']")
            print("使用方式示例: data = np.load('output.npz', allow_pickle=True)")
            print("获取嵌入列表: embeddings_list = data['embeddings']")
            print("获取第一条序列的嵌入矩阵: first_seq_embeddings = embeddings_list[0]")
            
        except Exception as e:
            print(f"处理数据集时出错: {e}")
            raise


def main():
    # 文件路径
    protvec_path = "data/protVec_100d_3grams.tsv"  # ProtVec向量文件路径
    input_file = "preprocess/seqsample/output/sampled_output.csv"  # 输入数据文件路径
    output_file = "preprocess/embedding/output/sampled_output_embeddings"  # 输出结果文件路径（无需扩展名）
    
    # 配置参数
    use_pentamers = True  # 是否使用五联体表示
    combine_method = "sum"  # 组合方法: "sum"或"mean"
    n_processes = min(32, max(1, mp.cpu_count() - 1))  # 进程数，不超过32
    batch_size = 8000  # 每批处理的数据量
    chunksize = 50  # 多进程任务分配的块大小
    
    # 初始化编码器
    encoder = ProteinEncoder(protvec_path)
    
    # 处理数据集
    encoder.process_protein_dataset_optimized(
        input_file=input_file,
        output_file=output_file,
        use_pentamers=use_pentamers,
        combine_method=combine_method,
        n_processes=n_processes,
        batch_size=batch_size,
        chunksize=chunksize
    )
    
    print("处理完成！")

if __name__ == "__main__":
    if mp.get_start_method(allow_none=True) is None:
        # 在Windows系统上设置多进程启动方法
        mp.set_start_method('spawn')  
    main()
