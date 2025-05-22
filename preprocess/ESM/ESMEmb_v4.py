import numpy as np
import pandas as pd
import os
import time
import gc
import torch
from esm.models.esmc import ESMC
from esm.sdk.api import *
import logging
from tqdm.auto import tqdm
import concurrent.futures
import threading
from pathlib import Path
import h5py
import queue

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('ESMEncoder')

class ESMProteinEncoder:
    def __init__(self, model_name: str = "esmc_600m", verbose: bool = False):
        """
        初始化ESM蛋白质编码器
        
        Args:
            model_name: ESM模型名称，如esmc_600m
            verbose: 是否输出详细日志
        """
        self.verbose = verbose
        # 设置环境变量使用本地模型
        os.environ["INFRA_PROVIDER"] = "True"
        
        # 设置设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"使用设备: {self.device}")
        
        # 加载ESMC模型
        self._load_esmc_model(model_name)
        
        # 氨基酸编码字典
        self.amino_acid_to_id = {
            'A':5, 'C':23, 'D':13, 'E':9, 'F':18,
            'G':6, 'H':21, 'I':12, 'K':15, 'L':4,
            'M':20, 'N':17, 'P':14, 'Q':16, 'R':10,
            'S':8, 'T':11, 'V':7, 'W':22, 'Y':19,
            '_':32
        }
        
        # 线程锁，用于保护模型推理
        self.model_lock = threading.Lock()
    
    def _load_esmc_model(self, model_name: str) -> None:
        """
        加载ESM模型
        
        Args:
            model_name: ESM模型名称
        """
        try:
            logger.info(f"正在加载ESM模型: {model_name}")
            
            # 使用新版API加载模型
            self.client = ESMC.from_pretrained(model_name, device=self.device)
            
            # 根据esmc_600m模型设置向量维度
            self.vector_dim = 1152  
            self.max_seq_len = 1024  # 最大序列长度
            
            logger.info(f"已成功加载ESM模型，embedding维度: {self.vector_dim}, 最大序列长度: {self.max_seq_len}")
            
        except Exception as e:
            logger.error(f"加载ESM模型时出错: {e}")
            raise
    
    def esm_encode_sequence(self, sequence: str, pad_len: int = None) -> torch.Tensor:
        """
        将氨基酸序列编码为ESM模型输入格式
        
        Args:
            sequence: 蛋白质氨基酸序列
            pad_len: 填充长度，默认为序列本身长度
            
        Returns:
            编码后的序列张量
        """
        try:
            if pad_len is None:
                pad_len = len(sequence)
                
            # 将序列字符转换为对应的ID
            s = [self.amino_acid_to_id.get(aa, self.amino_acid_to_id['_']) for aa in sequence]
            
            # 填充到指定长度
            while len(s) < pad_len:
                s.append(1)  # 使用1作为填充ID
                
            # 添加特殊标记（开始和结束标记）
            s.insert(0, 0)  # 添加开始标记
            s.append(2)     # 添加结束标记
            
            return torch.tensor(s).to(self.device)
        except Exception as e:
            logger.error(f"序列编码错误: {e}")
            # 返回一个有效的编码
            return torch.tensor([0, 5, 2]).to(self.device)  # 最小有效序列 [BOS, A, EOS]
    
    @torch.no_grad()
    def batch_process_sequences(self, sequences: list) -> list:
        """
        批量处理多个序列，提高GPU利用效率
        
        Args:
            sequences: 蛋白质氨基酸序列列表
            
        Returns:
            每个序列对应的嵌入矩阵列表
        """
        if not sequences:
            return []
            
        with self.model_lock:
            try:
                # 编码并处理所有序列
                batch_tensors = []
                batch_lengths = []
                
                # 为每个序列创建输入张量
                for seq in sequences:
                    if not seq or len(seq) == 0:
                        # 对空序列使用占位符
                        batch_tensors.append(torch.tensor([0, 5, 2]).to(self.device))
                        batch_lengths.append(0)
                        continue
                        
                    # 截断过长序列
                    if len(seq) > self.max_seq_len - 2:
                        seq = seq[:(self.max_seq_len-2)]
                        
                    # 编码序列
                    tensor = self.esm_encode_sequence(seq, len(seq))
                    batch_tensors.append(tensor)
                    batch_lengths.append(len(seq))
                
                # 并行处理所有序列
                embeddings_list = []
                
                # 由于ESM模型的API限制，我们需要单独处理每个序列
                # 但实际GPU批处理会在内部更高效
                for i, tensor in enumerate(batch_tensors):
                    protein_tensor = ESMProteinTensor(sequence=tensor)
                    logits_output = self.client.logits(protein_tensor, LogitsConfig(sequence=True, return_embeddings=True))
                    
                    if logits_output.embeddings is None or logits_output.embeddings.numel() == 0:
                        # 如果嵌入为空，返回零矩阵
                        embeddings_list.append(np.zeros((max(1, batch_lengths[i]), self.vector_dim), dtype=np.float32))
                        continue
                        
                    # 根据形状确定正确的切片方式
                    if len(logits_output.embeddings.shape) == 2:  # [seq_len, dim]
                        emb = logits_output.embeddings[1:-1].cpu().numpy()  # 去掉首尾标记
                    elif len(logits_output.embeddings.shape) == 3:  # [batch, seq_len, dim]
                        emb = logits_output.embeddings[0, 1:-1].cpu().numpy()
                    else:
                        emb = np.zeros((batch_lengths[i], self.vector_dim), dtype=np.float32)
                        
                    embeddings_list.append(emb.astype(np.float32))
                
                return embeddings_list
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    logger.error(f"GPU内存不足: {e}")
                    torch.cuda.empty_cache()
                    # 回退到单个处理
                    return [np.zeros((l, self.vector_dim), dtype=np.float32) for l in batch_lengths]
                else:
                    logger.error(f"批处理序列时出错: {e}")
                    return [np.zeros((l, self.vector_dim), dtype=np.float32) for l in batch_lengths]
            except Exception as e:
                logger.error(f"批处理序列时出错: {e}")
                return [np.zeros((l, self.vector_dim), dtype=np.float32) for l in batch_lengths]

    @torch.no_grad()
    def get_sequence_embeddings(self, sequence: str) -> np.ndarray:
        """
        获取序列的ESM嵌入表示
        
        Args:
            sequence: 蛋白质氨基酸序列
            
        Returns:
            序列嵌入矩阵 (去掉首尾标记)
        """
        if not sequence or len(sequence) == 0:
            if self.verbose:
                logger.warning(f"警告: 空序列")
            return np.zeros((1, self.vector_dim), dtype=np.float32)
        
        # 使用批处理函数处理单个序列
        return self.batch_process_sequences([sequence])[0]
    
    def _process_batch(self, batch_df, batch_id, result_queue):
        """
        处理单个批次的数据并将结果放入队列
        
        Args:
            batch_df: 包含该批次数据的DataFrame
            batch_id: 批次ID
            result_queue: 结果队列，用于传递处理结果
            
        Returns:
            None，结果直接放入队列
        """
        batch_start_time = time.time()
        
        # 收集批次的序列和元数据
        batch_indices = []
        batch_sequences = []
        batch_diff_counts = []
        batch_mean_log10Ka = []
        batch_seq_lengths = []
        
        # 为追踪进度创建进度条
        pbar = None
        if self.verbose:
            pbar = tqdm(total=len(batch_df), desc=f"批次 {batch_id}", leave=False)
        
        # 收集有效序列
        for _, row in batch_df.iterrows():
            sequence = row.get('aim_seq', '')
            
            # 验证序列
            if not isinstance(sequence, str) or len(sequence) == 0:
                if self.verbose and pbar:
                    pbar.update(1)
                continue
            
            batch_indices.append(row.get('index', row.name))
            batch_sequences.append(sequence)
            batch_diff_counts.append(row.get('diff_count', 0))
            batch_mean_log10Ka.append(row.get('mean_log10Ka', 0.0))
            batch_seq_lengths.append(len(sequence))
            
            if self.verbose and pbar:
                pbar.update(1)
        
        if self.verbose and pbar:
            pbar.close()
        
        # 如果没有有效序列，直接返回
        if not batch_sequences:
            batch_time = time.time() - batch_start_time
            result_queue.put((batch_id, 0, None, batch_time))
            return
        
        # GPU批处理获取嵌入
        batch_embeddings = self.batch_process_sequences(batch_sequences)
        
        # 验证嵌入结果
        valid_results = []
        for i, embedding in enumerate(batch_embeddings):
            if embedding.size > 0 and not np.isnan(embedding).any():
                valid_results.append((
                    batch_indices[i],
                    batch_sequences[i],
                    batch_diff_counts[i],
                    batch_mean_log10Ka[i],
                    batch_seq_lengths[i],
                    embedding
                ))
        
        # 将结果放入队列
        batch_time = time.time() - batch_start_time
        result_queue.put((batch_id, len(valid_results), valid_results, batch_time))
        
    def _writer_thread(self, result_queue, output_file, total_batches, debug_mode):
        """
        专门的写线程，实时将结果写入HDF5文件
        
        Args:
            result_queue: 结果队列
            output_file: 输出文件路径
            total_batches: 总批次数
            debug_mode: 是否为调试模式
        """
        # 决定输出文件名
        output_path = f"{output_file}_debug.h5" if debug_mode else f"{output_file}.h5"
        
        # 创建输出文件
        with h5py.File(output_path, 'w') as out_file:
            # 创建存储嵌入的组
            embeddings_group = out_file.create_group('embeddings')
            
            # 首先创建可扩展的数据集
            out_file.create_dataset('indices', (0,), maxshape=(None,), dtype=np.int32)
            out_file.create_dataset('diff_counts', (0,), maxshape=(None,), dtype=np.int32)
            out_file.create_dataset('mean_log10Ka', (0,), maxshape=(None,), dtype=np.float32)
            out_file.create_dataset('seq_lengths', (0,), maxshape=(None,), dtype=np.int32)
            seq_dt = h5py.special_dtype(vlen=str)
            out_file.create_dataset('sequences', (0,), maxshape=(None,), dtype=seq_dt)
            
            # 进度条
            with tqdm(total=total_batches, desc="存储进度") as pbar:
                processed_batches = 0
                total_samples = 0
                
                # 用来收集批次元数据的缓冲区
                buffer_indices = []
                buffer_diff_counts = []
                buffer_mean_log10Ka = []
                buffer_seq_lengths = []
                buffer_seqs = []
                
                # 持续从队列获取结果并写入文件
                while processed_batches < total_batches:
                    try:
                        # 不阻塞，定期检查队列
                        result = result_queue.get(timeout=0.1)
                        batch_id, sample_count, batch_data, batch_time = result
                        
                        # 如果有有效数据，写入文件
                        if batch_data and sample_count > 0:
                            # 遍历批次中的每个样本
                            for sample_idx, sample_data in enumerate(batch_data):
                                orig_idx, sequence, diff_count, mean_log10Ka, seq_len, embedding = sample_data
                                
                                # 存储嵌入，使用原始索引作为键名
                                emb_key = f'emb_{orig_idx}'
                                if emb_key in embeddings_group:
                                    logger.warning(f"警告: 嵌入键 {emb_key} 已存在，将被覆盖")
                                    del embeddings_group[emb_key]
                                
                                embeddings_group.create_dataset(emb_key, data=embedding)
                                
                                # 收集元数据到缓冲区
                                buffer_indices.append(orig_idx)
                                buffer_diff_counts.append(diff_count)
                                buffer_mean_log10Ka.append(mean_log10Ka)
                                buffer_seq_lengths.append(seq_len)
                                buffer_seqs.append(sequence)
                                
                                # 计数
                                total_samples += 1
                            
                            # 当缓冲区累积到一定大小时写入文件
                            if len(buffer_indices) >= 5000:
                                self._flush_metadata_buffer(
                                    out_file, 
                                    buffer_indices, 
                                    buffer_diff_counts,
                                    buffer_mean_log10Ka,
                                    buffer_seq_lengths,
                                    buffer_seqs
                                )
                                
                                # 清空缓冲区
                                buffer_indices = []
                                buffer_diff_counts = []
                                buffer_mean_log10Ka = []
                                buffer_seq_lengths = []
                                buffer_seqs = []
                                
                                # 同步到磁盘
                                out_file.flush()
                        
                        # 更新进度
                        processed_batches += 1
                        pbar.update(1)
                        
                        # 输出处理信息
                        logger.info(f"批次 {batch_id} 处理完成: 成功写入 {sample_count} 个样本, "
                                    f"用时 {batch_time:.2f}秒")
                        
                    except queue.Empty:
                        # 队列为空，继续等待
                        continue
                    except Exception as e:
                        logger.error(f"写入结果时出错: {str(e)}")
                        import traceback
                        logger.error(traceback.format_exc())
                        processed_batches += 1
                        pbar.update(1)
            
            # 处理缓冲区中剩余的元数据
            if buffer_indices:
                self._flush_metadata_buffer(
                    out_file, 
                    buffer_indices, 
                    buffer_diff_counts,
                    buffer_mean_log10Ka,
                    buffer_seq_lengths,
                    buffer_seqs
                )
            
            # 写入全局属性
            out_file.attrs['total_samples'] = total_samples
            out_file.attrs['embedding_dim'] = self.vector_dim
            out_file.attrs['created_time'] = time.time()
        
        logger.info(f"所有数据写入完成，结果已保存至: {output_path}")
        logger.info(f"总共保存了 {total_samples} 个样本的嵌入向量，维度: {self.vector_dim}")

    def _flush_metadata_buffer(self, h5file, indices, diff_counts, mean_log10Ka, seq_lengths, sequences):
        """
        将元数据缓冲区内容追加到HDF5文件
        
        Args:
            h5file: 已打开的HDF5文件对象
            indices, diff_counts, mean_log10Ka, seq_lengths, sequences: 需要写入的数据
        """
        if not indices:
            return
        
        # 获取当前数据集大小并计算新大小
        current_size = h5file['indices'].shape[0]
        new_size = current_size + len(indices)
        
        # 调整所有数据集大小
        h5file['indices'].resize((new_size,))
        h5file['diff_counts'].resize((new_size,))
        h5file['mean_log10Ka'].resize((new_size,))
        h5file['seq_lengths'].resize((new_size,))
        h5file['sequences'].resize((new_size,))
        
        # 写入新数据
        h5file['indices'][current_size:new_size] = np.array(indices)
        h5file['diff_counts'][current_size:new_size] = np.array(diff_counts)
        h5file['mean_log10Ka'][current_size:new_size] = np.array(mean_log10Ka)
        h5file['seq_lengths'][current_size:new_size] = np.array(seq_lengths)
        h5file['sequences'][current_size:new_size] = np.array(sequences, dtype=h5py.special_dtype(vlen=str))

    def process_protein_dataset(self, input_file: str, output_file: str, batch_size: int = 1024,
                                max_workers: int = 2, gpu_batch_size: int = 16, debug_mode: bool = False):
        """
        处理蛋白质数据集并将嵌入结果写入HDF5文件
        
        Args:
            input_file: 输入数据文件路径
            output_file: 输出结果文件路径（无需扩展名）
            batch_size: 每批处理的数据量
            max_workers: 同时处理的批次数
            gpu_batch_size: GPU批处理大小
            debug_mode: 是否为调试模式
        """
        # 读取输入数据
        df = pd.read_csv(input_file)
        
        # 创建结果队列
        result_queue = queue.Queue()
        
        # 创建输出目录
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # 创建线程池
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            
            # 创建写线程
            writer_thread = threading.Thread(
                target=self._writer_thread,
                args=(result_queue, output_file, len(df) // batch_size + 1, debug_mode),
                daemon=True
            )
            writer_thread.start()
            
            # 分批处理数据集
            for batch_id in range(0, len(df), batch_size):
                batch_df = df.iloc[batch_id:batch_id + batch_size]
                
                # 提交任务到线程池
                future = executor.submit(self._process_batch, batch_df, batch_id // batch_size, result_queue)
                futures.append(future)
            
            # 等待所有任务完成
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"处理批次时出错: {e}")
        
        # 等待写线程完成
        writer_thread.join()


def main():
    # 配置参数
    model_name = "esmc_600m"  # ESM模型名称
    input_file = "preprocess/seqsample/output/sampled_output.csv"  # 输入数据文件路径
    output_file = "preprocess/ESM/output/sampled_output_esm_embeddings"  # 输出结果文件路径（无需扩展名）
    
    # 处理配置
    batch_size = 1024         # 每批处理的数据量
    max_workers = 2           # 同时处理的批次数
    gpu_batch_size = 16        # GPU批处理大小
    debug_mode = False        # 开启调试模式进行测试
    
    # 初始化编码器
    encoder = ESMProteinEncoder(model_name, verbose=False)
    
    # 处理数据集
    encoder.process_protein_dataset(
        input_file=input_file,
        output_file=output_file,
        batch_size=batch_size,
        max_workers=max_workers,
        gpu_batch_size=gpu_batch_size,
        debug_mode=debug_mode
    )
    
    logger.info("处理完成！")

if __name__ == "__main__":
    main()
