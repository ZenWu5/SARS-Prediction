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
        
        # 使用锁保护模型推理，避免多线程访问GPU时的问题
        with self.model_lock:
            try:
                # 如果序列太长，截断它
                if len(sequence) > self.max_seq_len - 2:  # -2 是为了考虑特殊标记
                    if self.verbose:
                        logger.warning(f"警告: 序列已截断 ({len(sequence)} > {self.max_seq_len-2})")
                    sequence = sequence[:(self.max_seq_len-2)]
                
                # 创建蛋白质张量
                protein_tensor = ESMProteinTensor(sequence=self.esm_encode_sequence(sequence, len(sequence)))
                
                # 获取嵌入表示
                logits_output = self.client.logits(protein_tensor, LogitsConfig(sequence=True, return_embeddings=True))
                
                if logits_output.embeddings is None:
                    if self.verbose:
                        logger.warning("警告: 模型返回的embeddings为None")
                    return np.zeros((len(sequence), self.vector_dim), dtype=np.float32)
                
                # 移除特殊标记对应的嵌入 (首尾标记)
                embeddings = logits_output.embeddings
                
                # 检查embeddings形状，确保它不是空的
                if embeddings.numel() == 0:
                    if self.verbose:
                        logger.warning("警告: embeddings为空张量")
                    return np.zeros((len(sequence), self.vector_dim), dtype=np.float32)
                    
                # 根据形状确定正确的切片方式
                if len(embeddings.shape) == 2:  # [seq_len, dim]
                    sequence_embeddings = embeddings[1:-1].cpu().numpy()  # 去掉首尾标记
                elif len(embeddings.shape) == 3:  # [batch, seq_len, dim]
                    sequence_embeddings = embeddings[0, 1:-1].cpu().numpy()
                else:
                    if self.verbose:
                        logger.warning(f"警告: 不支持的embeddings形状: {embeddings.shape}")
                    return np.zeros((len(sequence), self.vector_dim), dtype=np.float32)
                    
                # 确保结果不是空矩阵
                if sequence_embeddings.shape[0] == 0:
                    if self.verbose:
                        logger.warning("警告: 提取后的嵌入为空")
                    return np.zeros((len(sequence), self.vector_dim), dtype=np.float32)
                    
                # 确保维度正确
                if sequence_embeddings.shape[1] != self.vector_dim:
                    if self.verbose:
                        logger.warning(f"警告: 嵌入维度({sequence_embeddings.shape[1]})与预期({self.vector_dim})不符")
                    self.vector_dim = sequence_embeddings.shape[1]
                    
                return sequence_embeddings.astype(np.float32)  # 确保类型为float32
                    
            except RuntimeError as e:
                if "out of memory" in str(e):
                    logger.error(f"GPU内存不足: {e}")
                    torch.cuda.empty_cache()
                    return np.zeros((len(sequence), self.vector_dim), dtype=np.float32)
                else:
                    logger.error(f"处理序列时出错: {e}")
                    return np.zeros((len(sequence), self.vector_dim), dtype=np.float32)
            except Exception as e:
                logger.error(f"处理序列时出错: {e}")
                return np.zeros((len(sequence), self.vector_dim), dtype=np.float32)

    def _process_batch(self, batch_df, batch_id, temp_dir):
        """
        处理单个批次的数据并保存中间结果
        
        Args:
            batch_df: 包含该批次数据的DataFrame
            batch_id: 批次ID
            temp_dir: 中间结果保存路径
            
        Returns:
            元组：(批次ID, 成功处理的样本数, 中间结果文件路径, 处理时间)
        """
        batch_start_time = time.time()
        
        # 为此批次创建结果列表
        batch_indices = []
        batch_sequences = []
        batch_diff_counts = []
        batch_mean_log10Ka = []
        batch_seq_lengths = []
        batch_embeddings = []
        
        successful_count = 0
        
        # 为了追踪进度，不在日志中显示
        pbar = None
        if self.verbose:
            pbar = tqdm(total=len(batch_df), desc=f"批次 {batch_id}", leave=False)
        
        # 处理批次中的每个样本
        for _, row in batch_df.iterrows():
            sequence = row.get('aim_seq', '')
            
            # 验证序列
            if not isinstance(sequence, str) or len(sequence) == 0:
                if self.verbose and pbar:
                    pbar.update(1)
                continue
            
            # 获取嵌入
            embedding_matrix = self.get_sequence_embeddings(sequence)
            
            # 验证嵌入矩阵
            if embedding_matrix.size > 0 and not np.isnan(embedding_matrix).any():
                batch_indices.append(row.get('index', row.name))
                batch_sequences.append(sequence)
                batch_diff_counts.append(row.get('diff_count', 0))
                batch_mean_log10Ka.append(row.get('mean_log10Ka', 0.0))
                batch_seq_lengths.append(len(sequence))
                batch_embeddings.append(embedding_matrix)
                successful_count += 1
            
            if self.verbose and pbar:
                pbar.update(1)
        
        if self.verbose and pbar:
            pbar.close()
        
        # 如果成功处理了数据，保存中间结果
        temp_file_path = None
        if successful_count > 0:
            temp_file_path = os.path.join(temp_dir, f"batch_{batch_id}.h5")
            
            # 使用HDF5格式保存中间结果，比NPZ更高效
            with h5py.File(temp_file_path, 'w') as f:
                # 创建组
                batch_group = f.create_group(f'batch_{batch_id}')
                
                # 保存索引和元数据
                batch_group.create_dataset('indices', data=np.array(batch_indices, dtype=np.int64))
                batch_group.create_dataset('diff_counts', data=np.array(batch_diff_counts, dtype=np.int32))
                batch_group.create_dataset('mean_log10Ka', data=np.array(batch_mean_log10Ka, dtype=np.float32))
                batch_group.create_dataset('seq_lengths', data=np.array(batch_seq_lengths, dtype=np.int32))
                
                # 保存序列 (特殊处理字符串类型)
                seq_dt = h5py.special_dtype(vlen=str)
                batch_group.create_dataset('sequences', data=np.array(batch_sequences, dtype=seq_dt))
                
                # 分别保存每个嵌入矩阵，避免内存问题
                embed_group = batch_group.create_group('embeddings')
                for i, emb in enumerate(batch_embeddings):
                    embed_group.create_dataset(f'emb_{i}', data=emb)
        
        batch_time = time.time() - batch_start_time
        return (batch_id, successful_count, temp_file_path, batch_time)
        
    def process_protein_dataset_optimized(self, 
                                       input_file: str, 
                                       output_file: str,
                                       batch_size: int = 32,
                                       max_workers: int = 1,
                                       debug_mode: bool = False):
        """
        异步批量处理蛋白质序列数据集
        
        Args:
            input_file: 输入CSV文件路径
            output_file: 输出NPZ或HDF5文件路径
            batch_size: 每批处理的数据量
            max_workers: 处理工作线程的最大数量 (注意：GPU推理仍是串行的)
            debug_mode: 是否开启调试模式
        """
        try:
            # 创建临时目录用于中间结果
            output_dir = os.path.dirname(output_file)
            temp_dir = os.path.join(output_dir, f"temp_embeddings_{int(time.time())}")
            os.makedirs(temp_dir, exist_ok=True)
            logger.info(f"创建临时目录: {temp_dir}")

            # 新增一行，确保目录存在
            if not os.path.exists(temp_dir):
                os.makedirs(temp_dir)
            logger.info(f"创建临时目录: {temp_dir}")
            
            # 验证输入文件
            try:
                sample_df = pd.read_csv(input_file, nrows=5)
                required_columns = ['aim_seq', 'diff_count', 'mean_log10Ka']
                missing_columns = [col for col in required_columns if col not in sample_df.columns]
                if missing_columns:
                    raise ValueError(f"输入文件缺少必要的列: {missing_columns}")
            except Exception as e:
                logger.error(f"读取输入文件时出错: {e}")
                raise
                
            # 获取总行数
            total_rows = sum(1 for _ in open(input_file)) - 1  # 减去标题行
            logger.info(f"数据集共有 {total_rows} 条记录")
            
            # 在调试模式下只处理少量数据
            if debug_mode:
                max_rows = min(50, total_rows)
                logger.info(f"调试模式: 只处理前 {max_rows} 条记录")
                df_full = pd.read_csv(input_file, nrows=max_rows)
                total_batches = (len(df_full) + batch_size - 1) // batch_size
                total_to_process = len(df_full)
            else:
                # 读取所有数据到内存 (如果数据集太大，这里可以使用chunksize迭代读取)
                df_full = pd.read_csv(input_file)
                total_batches = (len(df_full) + batch_size - 1) // batch_size
                total_to_process = len(df_full)
                
            logger.info(f"共划分为 {total_batches} 个批次")
            
            # 准备批次
            batch_list = []
            for i in range(0, total_to_process, batch_size):
                end = min(i + batch_size, total_to_process)
                batch_list.append((df_full.iloc[i:end], i // batch_size + 1))
            
            # 结果列表
            batch_results = []
            
            # 使用线程池处理批次
            with tqdm(total=total_to_process, desc="总进度") as pbar:
                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                    # 提交批次处理任务
                    future_to_batch = {
                        executor.submit(self._process_batch, batch_df, batch_id, temp_dir): (batch_id, len(batch_df))
                        for batch_df, batch_id in batch_list
                    }
                    
                    # 收集结果
                    for future in concurrent.futures.as_completed(future_to_batch):
                        batch_id, batch_size = future_to_batch[future]
                        try:
                            result = future.result()
                            batch_results.append(result)
                            
                            # 更新进度条和日志
                            pbar.update(batch_size)
                            
                            logger.info(f"批次 {result[0]} 完成: 成功 {result[1]}/{batch_size} 样本, "
                                     f"用时 {result[3]:.2f}秒")
                            
                            # 清理内存
                            gc.collect()
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                                
                        except Exception as exc:
                            logger.error(f"批次 {batch_id} 处理出错: {exc}")
                            pbar.update(batch_size)
            
            # 检查是否有成功的批次
            successful_batches = [r for r in batch_results if r[2] is not None]
            if not successful_batches:
                logger.warning("没有成功处理任何批次，无结果可保存")
                return
            
            # 总结处理结果
            total_successful = sum(r[1] for r in batch_results)
            logger.info(f"所有批次处理完成。成功处理: {total_successful}/{total_to_process} 样本")
            
            # 现在合并所有中间结果
            self._merge_intermediate_results(successful_batches, output_file, temp_dir, debug_mode)
            
        except Exception as e:
            logger.error(f"处理数据集时出错: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise
        finally:
            # 如果不是调试模式，清理临时文件
            if not debug_mode and os.path.exists(temp_dir):
                try:
                    import shutil
                    shutil.rmtree(temp_dir)
                    logger.info(f"已清理临时目录: {temp_dir}")
                except Exception as e:
                    logger.warning(f"清理临时目录时出错: {e}")
    

    def _merge_intermediate_results(self, batch_results, output_file, temp_dir, debug_mode):
        """
        合并所有中间 HDF5 文件，仅保留 embeddings, mean_log10Ka, diff_count，编号为 emb_{i}
        """
        logger.info("开始合并中间结果...")

        sorted_batches = sorted(batch_results, key=lambda x: x[0])
        temp_files = [r[2] for r in sorted_batches if r[2] is not None]

        mean_log10Ka_list = []
        diff_count_list = []
        embedding_info = []

        use_hdf5 = True
        output_path = f"{output_file}_debug.h5" if debug_mode else f"{output_file}.h5"
        total_samples = 0

        if use_hdf5:
            with h5py.File(output_path, 'w') as out_file:
                embeddings_group = out_file.create_group('embeddings')

                for temp_file in temp_files:
                    with h5py.File(temp_file, 'r') as in_file:
                        batch_group = in_file[list(in_file.keys())[0]]

                        mean_vals   = batch_group['mean_log10Ka'][:].astype(np.float32)
                        diff_counts = batch_group['diff_counts'][:].astype(np.int16)
                        emb_group   = batch_group['embeddings']

                        mean_log10Ka_list.append(mean_vals)
                        diff_count_list.append(diff_counts)

                        for i in range(len(mean_vals)):
                            arr = emb_group[f'emb_{i}'][:].astype(np.float32)
                            embeddings_group.create_dataset(f'emb_{total_samples}', data=arr)
                            total_samples += 1

                out_file.create_dataset('mean_log10Ka', data=np.concatenate(mean_log10Ka_list))
                out_file.create_dataset('diff_count', data=np.concatenate(diff_count_list))

                out_file.attrs['total_samples'] = total_samples
                out_file.attrs['embedding_dim'] = self.vector_dim
                out_file.attrs['created_time']  = time.time()

            logger.info(f"合并完成，结果已保存至: {output_path}")
            logger.info(f"总共保存了 {total_samples} 个样本的嵌入向量，维度: {self.vector_dim}")    
    
def main():
    # 配置参数
    model_name = "esmc_600m"  # ESM模型名称
    input_file = "preprocess/seqsample/output/sampled_output.csv"  # 输入数据文件路径
    output_file = "preprocess/ESM/output/sampled_output_esm_embeddings"  # 输出结果文件路径（无需扩展名）
    
    # 处理配置
    batch_size = 1024  # 每批处理的数据量
    max_workers = 1   # 同时处理的批次数（注意：实际GPU处理仍是串行的）
    debug_mode = False  # 开启调试模式进行测试
    
    # 初始化编码器 (verbose=False 减少输出)
    encoder = ESMProteinEncoder(model_name, verbose=False)
    
    # 处理数据集
    encoder.process_protein_dataset_optimized(
        input_file=input_file,
        output_file=output_file,
        batch_size=batch_size,
        max_workers=max_workers,
        debug_mode=debug_mode
    )
    
    logger.info("处理完成！")

if __name__ == "__main__":
    main()
