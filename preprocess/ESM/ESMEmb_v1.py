import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Union, Any, Optional
import os
import time
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
import gc
import torch
from esm.models.esmc import ESMC
from esm.sdk.api import *
from esm.tokenization import EsmSequenceTokenizer

class ESMProteinEncoder:
    def __init__(self, model_name: str = "esmc_600m"):
        """
        初始化ESM蛋白质编码器
        
        Args:
            model_name: ESM模型名称或路径
        """
        # 设置环境变量以使用本地模型
        os.environ["INFRA_PROVIDER"] = "True"
        
        # 加载ESM模型
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")
        
        # 加载ESMC模型
        self._load_esmc_model(model_name)
        
        # ESM编码字典
        self.amino_acid_to_id = {
            'A':5, 'C':23, 'D':13, 'E':9, 'F':18,
            'G':6, 'H':21, 'I':12, 'K':15, 'L':4,
            'M':20, 'N':17, 'P':14, 'Q':16, 'R':10,
            'S':8, 'T':11, 'V':7, 'W':22, 'Y':19,
            '_':32
        }
    
    def _load_esmc_model(self, model_name: str) -> None:
        """
        加载ESM模型
        
        Args:
            model_name: ESM模型名称或路径
        """
        try:
            print(f"正在加载ESM模型: {model_name}")
            
            # 使用新版API加载模型
            self.client = ESMC.from_pretrained(model_name, device=self.device)
            
            # 设置向量维度和最大序列长度
            # 对于esmc_600m模型，embedding维度是1280
            self.vector_dim = 1280  # 可能需要根据具体模型调整
            self.max_seq_len = 1024  # ESM-2 和 ESMC 的最大序列长度通常为1024
            
            print(f"已成功加载ESM模型，embedding维度: {self.vector_dim}")
            print(f"最大序列长度: {self.max_seq_len}")
            
        except Exception as e:
            print(f"加载ESM模型时出错: {e}")
            raise
    
    def esm_encode_sequence(self, sequence: str) -> torch.Tensor:
        """
        将氨基酸序列编码为ESM模型输入格式
        
        Args:
            sequence: 蛋白质氨基酸序列
            
        Returns:
            编码后的序列张量
        """
        # 将序列字符转换为对应的ID
        encoded = [self.amino_acid_to_id.get(aa, self.amino_acid_to_id['_']) for aa in sequence]
        
        # 添加特殊标记（开始和结束标记）
        encoded = [0] + encoded + [2]
        
        return torch.tensor(encoded).to(self.device)
    
    @torch.no_grad()
    def get_sequence_embeddings(self, sequence: str) -> np.ndarray:
        """
        获取序列的ESM嵌入表示
        
        Args:
            sequence: 蛋白质氨基酸序列
            
        Returns:
            序列嵌入矩阵
        """
        try:
            # 如果序列太长，截断它
            if len(sequence) > self.max_seq_len - 2:  # -2 是为了考虑特殊标记
                print(f"警告: 序列已截断 ({len(sequence)} > {self.max_seq_len-2})")
                sequence = sequence[:(self.max_seq_len-2)]
            
            # 创建蛋白质张量
            protein_tensor = ESMProteinTensor(sequence=self.esm_encode_sequence(sequence))
            
            # 获取嵌入表示
            logits_output = self.client.logits(protein_tensor, LogitsConfig(sequence=True, return_embeddings=True))
            embeddings = logits_output.embeddings
            
            # 移除特殊标记对应的嵌入 (CLS和EOS标记)
            sequence_embeddings = embeddings[1:-1].cpu().numpy()
            
            return sequence_embeddings
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"GPU内存不足，尝试在CPU上处理...")
                # 清理GPU缓存
                torch.cuda.empty_cache()
                
                # 在CPU上重试
                old_device = self.device
                self.device = torch.device("cpu")
                self.client = self.client.to(self.device)
                
                # 重新编码并获取嵌入
                protein_tensor = ESMProteinTensor(sequence=self.esm_encode_sequence(sequence))
                logits_output = self.client.logits(protein_tensor, LogitsConfig(sequence=True, return_embeddings=True))
                embeddings = logits_output.embeddings
                sequence_embeddings = embeddings[1:-1].cpu().numpy()
                
                # 将模型移回原始设备
                self.device = old_device
                self.client = self.client.to(self.device)
                
                return sequence_embeddings
            else:
                print(f"处理序列时出错: {e}")
                return np.zeros((len(sequence), self.vector_dim), dtype=np.float32)
        except Exception as e:
            print(f"处理序列时出错: {e}")
            return np.zeros((len(sequence), self.vector_dim), dtype=np.float32)

    def _process_sequence_efficiently(self, 
                                    row_tuple: Tuple[int, pd.Series]) -> Tuple[int, np.ndarray]:
        """
        处理单个序列的辅助函数
        
        Args:
            row_tuple: 行索引和序列数据的元组
            
        Returns:
            行索引和嵌入表示的元组
        """
        idx, row = row_tuple
        sequence = row['aim_seq']
        
        try:
            # 获取ESM嵌入
            embeddings = self.get_sequence_embeddings(sequence)
            return idx, embeddings
                
        except Exception as e:
            print(f"处理序列时出错 (索引 {idx}): {e}")
            return idx, np.zeros((len(sequence), self.vector_dim), dtype=np.float32)

    def process_protein_dataset_optimized(self, 
                                        input_file: str, 
                                        output_file: str, 
                                        n_processes: int = 1,
                                        batch_size: int = 100,
                                        chunksize: int = 1):
        """
        批量处理蛋白质序列数据集
        
        Args:
            input_file: 输入CSV文件路径
            output_file: 输出NPZ文件路径，无需扩展名
            n_processes: 进程数，使用GPU时建议为1
            batch_size: 每批处理的数据量
            chunksize: 多进程任务分配的块大小
        """
        try:
            # 如果使用GPU，设置进程数为1
            if torch.cuda.is_available() and n_processes > 1:
                print("使用GPU时建议进程数为1，已自动调整")
                n_processes = 1
                
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
            
            # 设置多进程或单进程处理
            if n_processes > 1:
                ctx = mp.get_context('spawn')  # 使用spawn方式，更安全的多进程方法
                with ctx.Pool(processes=n_processes) as pool:
                    # 按批次处理数据
                    for chunk_df in pd.read_csv(input_file, chunksize=batch_size):
                        batch_start_time = time.time()
                        print(f"\n处理批次 {batch_num}, 大小: {len(chunk_df)} 行")
                        
                        # 准备批次数据
                        row_tuples = list(chunk_df.iterrows())
                        
                        # 使用imap_unordered提高性能并设置合适的chunksize
                        results = list(tqdm(
                            pool.imap_unordered(self._process_sequence_efficiently, row_tuples, chunksize=chunksize),
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
            else:
                # 单进程处理（GPU环境推荐）
                for chunk_df in pd.read_csv(input_file, chunksize=batch_size):
                    batch_start_time = time.time()
                    print(f"\n处理批次 {batch_num}, 大小: {len(chunk_df)} 行")
                    
                    # 对每行数据进行处理
                    for idx, row in tqdm(chunk_df.iterrows(), total=len(chunk_df), 
                                        desc=f"批次 {batch_num}/{(total_rows + batch_size - 1) // batch_size}"):
                        sequence = row['aim_seq']
                        embedding_matrix = self.get_sequence_embeddings(sequence)
                        
                        if embedding_matrix.size > 0:
                            original_indices.append(row.get('index', idx))
                            sequences.append(sequence)
                            diff_counts.append(row['diff_count'])
                            mean_log10Ka_values.append(row['mean_log10Ka'])
                            embeddings_list.append(embedding_matrix)
                            seq_lengths.append(len(sequence))
                    
                    # 清理内存
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
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
    # 配置参数
    model_name = "esmc_600m"  # ESM模型名称
    input_file = "preprocess/seqsample/output/sampled_output.csv"  # 输入数据文件路径
    output_file = "preprocess/ESM/output/sampled_output_esm_embeddings"  # 输出结果文件路径（无需扩展名）
    
    # 进程和批处理配置
    n_processes = 2  # 使用GPU时建议为1
    batch_size = 512  # 每批处理的数据量
    chunksize = 1  # 多进程任务分配的块大小
    
    # 初始化编码器
    encoder = ESMProteinEncoder(model_name)
    
    # 处理数据集
    encoder.process_protein_dataset_optimized(
        input_file=input_file,
        output_file=output_file,
        n_processes=n_processes,
        batch_size=batch_size,
        chunksize=chunksize
    )
    
    print("处理完成！")

if __name__ == "__main__":
    # 使用spawn方法以避免CUDA在多进程中的问题
    mp.set_start_method('spawn', force=True)
    main()
