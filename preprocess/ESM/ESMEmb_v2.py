import numpy as np
import pandas as pd
import os
import time
from tqdm import tqdm
import gc
import torch
from esm.models.esmc import ESMC
from esm.sdk.api import *
import logging

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
                
            return sequence_embeddings
                
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

    def process_protein_dataset_optimized(self, 
                                        input_file: str, 
                                        output_file: str,
                                        batch_size: int = 32,
                                        debug_mode: bool = False):
        """
        批量处理蛋白质序列数据集
        
        Args:
            input_file: 输入CSV文件路径
            output_file: 输出NPZ文件路径，无需扩展名
            batch_size: 每批处理的数据量
            debug_mode: 是否开启调试模式
        """
        try:
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
            logger.info(f"数据集共有 {total_rows} 条记录，将分批处理")
            
            # 在调试模式下只处理少量数据
            if debug_mode:
                max_rows = min(5, total_rows)
                logger.info(f"调试模式: 只处理前 {max_rows} 条记录")
                dfs_to_process = [pd.read_csv(input_file, nrows=max_rows)]
                total_to_process = max_rows
            else:
                dfs_to_process = pd.read_csv(input_file, chunksize=batch_size)
                total_to_process = total_rows
            
            # 创建输出目录
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            # 初始化数据存储
            original_indices = []
            sequences = []
            diff_counts = []
            mean_log10Ka_values = []
            embeddings_list = []
            seq_lengths = []
            
            # 处理计数器
            batch_num = 1
            successful_count = 0
            
            # 总进度条
            total_batches = (total_to_process + batch_size - 1) // batch_size
            overall_pbar = tqdm(total=total_to_process, desc="总进度", position=0)
            
            # 处理每个批次
            for chunk_df in dfs_to_process:
                batch_start_time = time.time()
                
                if self.verbose:
                    logger.info(f"处理批次 {batch_num}/{total_batches}, 大小: {len(chunk_df)} 行")
                
                # 处理每行数据
                batch_successful = 0
                for idx, row in enumerate(chunk_df.itertuples()):
                    sequence = getattr(row, 'aim_seq', '')
                    
                    # 验证序列
                    if not isinstance(sequence, str) or len(sequence) == 0:
                        if self.verbose:
                            logger.warning(f"警告: 索引 {idx} 的序列无效")
                        continue
                    
                    # 在调试模式下打印序列
                    if debug_mode and self.verbose:
                        logger.debug(f"处理序列(索引 {idx}): {sequence[:50]}... (长度: {len(sequence)})")
                    
                    # 获取嵌入
                    embedding_matrix = self.get_sequence_embeddings(sequence)
                    
                    # 验证嵌入矩阵
                    if embedding_matrix.size > 0 and not np.isnan(embedding_matrix).any():
                        original_indices.append(getattr(row, 'Index', idx))
                        sequences.append(sequence)
                        diff_counts.append(getattr(row, 'diff_count', 0))
                        mean_log10Ka_values.append(getattr(row, 'mean_log10Ka', 0.0))
                        embeddings_list.append(embedding_matrix)
                        seq_lengths.append(len(sequence))
                        successful_count += 1
                        batch_successful += 1
                        
                        if debug_mode and self.verbose:
                            logger.debug(f"成功获取嵌入: 形状 {embedding_matrix.shape}")
                    
                    # 更新总进度
                    overall_pbar.update(1)
                
                # 批次处理完成
                batch_end_time = time.time()
                batch_time = batch_end_time - batch_start_time
                
                if self.verbose or batch_num %10 == 0:
                    logger.info(f"批次 {batch_num}/{total_batches} 完成，用时: {batch_time:.2f}秒, 成功率: {batch_successful}/{len(chunk_df)}")
                
                # 每10个批次显示一次总体成功率
                if batch_num % 10 == 0:
                    logger.info(f"总成功率: {successful_count}/{overall_pbar.n} ({successful_count/overall_pbar.n*100:.2f}%)")
                
                # 每20个批次保存中间结果
                if not debug_mode and batch_num % 20 == 0 and len(embeddings_list) > 0:
                    temp_output = f"{output_file}_batch_{batch_num}.npz"
                    np.savez_compressed(
                        temp_output,
                        indices=np.array(original_indices, dtype=np.int64),
                        sequences=np.array(sequences, dtype=object),
                        diff_counts=np.array(diff_counts, dtype=np.int32),
                        mean_log10Ka=np.array(mean_log10Ka_values, dtype=np.float32),
                        seq_lengths=np.array(seq_lengths, dtype=np.int32),
                        embeddings=embeddings_list
                    )
                    logger.info(f"已保存中间结果至: {temp_output}")
                
                # 清理内存
                gc.collect()
                torch.cuda.empty_cache()
                
                batch_num += 1
                
                # 调试模式下处理一个批次后退出
                if debug_mode:
                    break
            
            # 关闭进度条
            overall_pbar.close()
            
            # 确保有数据要保存
            if len(embeddings_list) == 0:
                logger.warning("警告: 没有成功处理任何序列，无法保存结果")
                return
                
            # 准备保存
            original_indices = np.array(original_indices, dtype=np.int64)
            sequences = np.array(sequences, dtype=object)
            diff_counts = np.array(diff_counts, dtype=np.int32)
            mean_log10Ka_values = np.array(mean_log10Ka_values, dtype=np.float32)
            seq_lengths = np.array(seq_lengths, dtype=np.int32)
            
            # 保存最终结果
            output_filename = f"{output_file}_debug.npz" if debug_mode else f"{output_file}.npz"
            np.savez_compressed(
                output_filename,
                indices=original_indices,
                sequences=sequences,
                diff_counts=diff_counts,
                mean_log10Ka=mean_log10Ka_values,
                seq_lengths=seq_lengths,
                embeddings=embeddings_list
            )
            
            # 输出结果
            logger.info(f"处理完成，结果已保存至: {output_filename}")
            logger.info(f"总共成功处理: {successful_count}/{total_to_process} 条序列")
            
            if len(embeddings_list) > 0:
                logger.info(f"嵌入矩阵示例形状: {embeddings_list[0].shape}")
                logger.info(f"嵌入维度: {self.vector_dim}")
            
        except Exception as e:
            logger.error(f"处理数据集时出错: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise


def main():
    # 配置参数
    model_name = "esmc_600m"  # ESM模型名称
    input_file = "preprocess/seqsample/output/sampled_output.csv"  # 输入数据文件路径
    output_file = "preprocess/ESM/output/sampled_output_esm_embeddings"  # 输出结果文件路径（无需扩展名）
    
    # 处理配置
    batch_size = 256  # 每批处理的数据量
    debug_mode = False  # 开启调试模式进行测试
    
    # 初始化编码器 (verbose=False 减少输出)
    encoder = ESMProteinEncoder(model_name, verbose=False)
    
    # 处理数据集
    encoder.process_protein_dataset_optimized(
        input_file=input_file,
        output_file=output_file,
        batch_size=batch_size,
        debug_mode=debug_mode
    )
    
    logger.info("处理完成！")

if __name__ == "__main__":
    main()
