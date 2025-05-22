import os
import time
import multiprocessing
import pandas as pd
import numpy as np
from tqdm import tqdm
from functools import partial
import tempfile
import psutil
import logging
from concurrent.futures import ProcessPoolExecutor
import sys

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("sequence_processing.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

# 可配置参数 - 直接在脚本中修改这些值
INPUT_FILE = r"preprocess\seqsample\output\aim_seq_mean_log10Ka.csv"  # 输入文件路径
OUTPUT_FILE = r"preprocess\seqsample\output\sampled_output_all.csv"         # 输出文件路径
REFERENCE_SEQUENCE = "NITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYFPLQSYGFQPTNGVGYQPYRVVVLSFELLHAPATVCGPKKST"
REF_INDEX = 0                       # 参考序列的索引(如不提供参考序列)
THRESHOLD = None                    # 差异阈值，只保留差异数量≤此值的序列
CHUNK_SIZE = 100000                  # 每次处理的序列数量
WORKERS = None                      # 工作进程数，默认为CPU核心数减1
TEMP_DIR = None                     # 临时文件目录，默认为系统临时目录

# 序列列名，按优先顺序列出
SEQUENCE_COLUMN_NAMES = ['aim_seq', 'sequence', 'seq', 'amino_acid_sequence']

def memory_usage():
    """获取当前内存使用情况"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # 转换为MB

def debug_columns(df, max_rows=5):
    """输出数据帧的列名和前几行，用于调试"""
    logger.info(f"数据帧列名: {df.columns.tolist()}")
    if len(df) > 0:
        logger.info(f"第一行样例: {df.iloc[0].to_dict()}")
    return df

def read_data_chunk(file_path, chunk_size, skip_rows=0):
    """分块读取大型文件，自动跳过标题行和BOM"""
    try:
        # 尝试读取文件头以确定分隔符和列名
        if skip_rows == 0:
            header_sample = pd.read_csv(
                file_path, 
                nrows=1, 
                encoding='utf-8-sig',  # 处理BOM
                engine='python'
            )
            logger.info(f"检测到文件列名: {header_sample.columns.tolist()}")
            
            # 确认序列列存在
            seq_col = None
            for col_name in SEQUENCE_COLUMN_NAMES:
                if col_name in header_sample.columns:
                    seq_col = col_name
                    break
                    
            if seq_col is None:
                logger.warning(f"未找到序列列! 可用列: {header_sample.columns.tolist()}")
                logger.warning("尝试使用第一个非索引列作为序列列")

        # 读取数据块
        df_iter = pd.read_csv(
            file_path,
            sep=',',                # 明确指定分隔符
            skiprows=skip_rows + (1 if skip_rows > 0 else 0),  # 首次读取保留标题行
            nrows=chunk_size,
            encoding='utf-8-sig',   # 处理BOM
            engine='python'
        )
        
        if df_iter.empty:
            return None
            
        # 添加index列
        if 'index' not in df_iter.columns:
            df_iter.insert(0, 'index', range(skip_rows, skip_rows + len(df_iter)))
        
        # 返回前调试列信息
        if skip_rows == 0:
            debug_columns(df_iter)
        
        return df_iter
    except Exception as e:
        logger.error(f"读取数据块时出错: {e}")
        return None

def compare_sequences(seq1, seq2):
    """比较两个序列，返回差异数量和位置信息"""
    if len(seq1) != len(seq2):
        logger.debug(f"序列长度不匹配: {len(seq1)} vs {len(seq2)}")
    
    # 找出所有差异
    differences = []
    diff_count = 0
    
    min_len = min(len(seq1), len(seq2))
    for i in range(min_len):
        if seq1[i] != seq2[i]:
            diff_count += 1
            # 记录变化格式：原始氨基酸+位置+新氨基酸
            differences.append(f"{seq1[i]}{i+1}{seq2[i]}")
    
    # 处理长度差异
    if len(seq1) < len(seq2):
        for i in range(len(seq1), len(seq2)):
            diff_count += 1
            differences.append(f"-{i+1}{seq2[i]}")  # 插入
    elif len(seq1) > len(seq2):
        for i in range(len(seq2), len(seq1)):
            diff_count += 1
            differences.append(f"{seq1[i]}{i+1}-")  # 删除
            
    return diff_count, differences

def get_sequence_from_row(row):
    """从行数据中获取序列，尝试不同的列名"""
    # 按优先顺序尝试不同列名
    for col_name in SEQUENCE_COLUMN_NAMES:
        if col_name in row and pd.notna(row[col_name]) and row[col_name]:
            return row[col_name]
    
    # 如果没有找到序列列，尝试使用第一个非索引、非空的列
    for col in row.index:
        if col != 'index' and not col.startswith('Unnamed') and pd.notna(row[col]) and isinstance(row[col], str):
            if len(row[col]) > 10:  # 假设序列至少有10个字符
                return row[col]
    
    # 没有找到任何可用序列
    logger.warning(f"在索引 {row.get('index', 'unknown')} 的行中找不到序列")
    return None

def process_sequence(row, ref_seq):
    try:
        seq = get_sequence_from_row(row)
        if seq is None:
            return None

        diff_count, differences = compare_sequences(ref_seq, seq)

        # 只保留你需要的字段
        result = {
            'index': row.get('index', 0),
            'aim_seq': seq,  # 用实际的列名
            'diff_count': diff_count,
            'differences': ';'.join(differences)
        }
        # 可选：如果你还需要其它固定字段，手动加上
        for col in ['mean_log10Ka', 'count']:
            if col in row:
                result[col] = row[col]
        return result
    except Exception as e:
        logger.error(f"处理序列时出错 (索引 {row.get('index', 'unknown')}): {str(e)}")
        return None

def process_chunk(chunk_data, ref_seq, threshold=None):
    """处理数据块"""
    results = []
    
    for _, row in chunk_data.iterrows():
        result = process_sequence(row, ref_seq)
        if result is not None:
            # 应用阈值过滤
            if threshold is None or result['diff_count'] <= threshold:
                results.append(result)
    
    return results

def process_chunk_parallel(chunk_data, ref_seq, threshold=None, n_workers=None):
    """并行处理数据块"""
    if n_workers is None:
        n_workers = max(1, multiprocessing.cpu_count() - 1)
    
    # 准备工作函数
    worker_fn = partial(process_sequence, ref_seq=ref_seq)
    results = []
    
    try:
        # 使用ProcessPoolExecutor进行并行处理
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            # 提交所有任务
            future_results = list(executor.map(worker_fn, [row for _, row in chunk_data.iterrows()]))
            
            # 收集结果并应用阈值过滤
            for result in future_results:
                if result is not None:
                    if threshold is None or result['diff_count'] <= threshold:
                        results.append(result)
    except KeyboardInterrupt:
        logger.warning("检测到键盘中断，正在尝试优雅地退出...")
        # 在这里可以添加清理代码
        # 返回已处理的结果
        return results
    except Exception as e:
        logger.error(f"并行处理时出错: {str(e)}")
        # 返回已收集的结果
        return results
    
    return results

def main():
    # 设置工作进程数
    global WORKERS
    if WORKERS is None:
        WORKERS = max(1, multiprocessing.cpu_count() - 1)
    
    logger.info(f"开始处理数据。使用 {WORKERS} 个工作进程.")
    start_time = time.time()
    
    # 确定参考序列
    reference_sequence = REFERENCE_SEQUENCE
    if reference_sequence is None:
        # 从输入文件中获取参考序列
        first_chunk = read_data_chunk(INPUT_FILE, 1)
        if first_chunk is None or first_chunk.empty:
            logger.error("无法读取数据文件")
            return
        
        ref_row = first_chunk.iloc[REF_INDEX]
        reference_sequence = get_sequence_from_row(ref_row)
        if reference_sequence is None:
            logger.error("无法从参考行中提取序列，请检查文件格式")
            return
            
        
    # 设置临时目录
    temp_dir = TEMP_DIR if TEMP_DIR else tempfile.mkdtemp()
    os.makedirs(temp_dir, exist_ok=True)
    logger.info(f"使用临时目录: {temp_dir}")
    
    # 统计文件行数以显示进度
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8-sig') as f:
            total_lines = sum(1 for _ in f)
        logger.info(f"数据集总行数: {total_lines}")
    except:
        total_lines = None
        logger.warning("无法统计总行数，将不显示精确进度")
    
    # 处理数据
    processed_rows = 0
    temp_files = []
    
    try:
        with tqdm(total=total_lines, desc="处理序列") as pbar:
            while True:
                # 读取下一个数据块
                chunk = read_data_chunk(INPUT_FILE, CHUNK_SIZE, processed_rows)
                if chunk is None or chunk.empty:
                    break
                
                chunk_size = len(chunk)
                logger.debug(f"读取数据块: {processed_rows} 到 {processed_rows + chunk_size}")
                
                # 记录内存使用
                mem_before = memory_usage()
                
                # 并行处理数据块
                results = process_chunk_parallel(
                    chunk, reference_sequence, THRESHOLD, WORKERS
                )
                
                # 将结果写入临时文件
                if results:
                    temp_file = os.path.join(temp_dir, f"temp_results_{processed_rows}.csv")
                    pd.DataFrame(results).to_csv(temp_file, index=False)
                    temp_files.append(temp_file)
                    
                    logger.debug(f"写入临时文件: {temp_file}, 结果数: {len(results)}")
                
                # 更新进度
                processed_rows += chunk_size
                pbar.update(chunk_size)
                
                # 记录并显示内存使用
                mem_after = memory_usage()
                logger.debug(f"内存使用: {mem_before:.1f} MB -> {mem_after:.1f} MB")
    
    except KeyboardInterrupt:
        logger.warning("检测到键盘中断，正在完成当前处理...")
    except Exception as e:
        logger.error(f"处理数据时出错: {str(e)}")
    finally:
        # 合并所有临时文件
        if temp_files:
            logger.info(f"合并 {len(temp_files)} 个结果文件...")
            
            # 创建输出目录
            os.makedirs(os.path.dirname(os.path.abspath(OUTPUT_FILE)), exist_ok=True)
            
            with open(OUTPUT_FILE, 'w', encoding='utf-8') as outfile:
                # 标题行
                header_written = False
                
                # 合并所有临时文件
                for temp_file in temp_files:
                    try:
                        temp_df = pd.read_csv(temp_file)
                        
                        if not header_written:
                            # 写入标题行
                            outfile.write(','.join(temp_df.columns) + '\n')
                            header_written = True
                        
                        # 写入数据
                        temp_df.to_csv(outfile, index=False, header=False, mode='a',lineterminator='\n')
                        
                        # 删除临时文件
                        os.remove(temp_file)
                    except Exception as e:
                        logger.error(f"处理临时文件 {temp_file} 时出错: {str(e)}")
            
            # 尝试删除临时目录
            try:
                if TEMP_DIR is None:  # 只删除我们创建的默认临时目录
                    os.rmdir(temp_dir)
            except:
                logger.warning(f"无法删除临时目录: {temp_dir}")
        
        # 计算和显示统计信息
        try:
            if os.path.exists(OUTPUT_FILE):
                # 只读取头部获取行数
                with open(OUTPUT_FILE, 'r', encoding='utf-8', newline="") as f:
                    line_count = sum(1 for _ in f) - 1  # 减去标题行
                
                logger.info(f"处理完成. 总共处理 {processed_rows} 条序列，输出 {line_count} 条结果")
                
                # 可选：对结果进行采样统计
                if line_count > 0:
                    sample_size = min(1000, line_count)
                    sample_df = pd.read_csv(OUTPUT_FILE, nrows=sample_size)
                    if 'diff_count' in sample_df.columns:
                        logger.info(f"差异统计 (基于 {sample_size} 条样本): 最小={sample_df['diff_count'].min()}, "
                                  f"最大={sample_df['diff_count'].max()}, "
                                  f"平均={sample_df['diff_count'].mean():.2f}, "
                                  f"中位数={sample_df['diff_count'].median()}")
            else:
                logger.warning(f"输出文件 {OUTPUT_FILE} 未创建，处理可能未完成")
        except Exception as e:
            logger.error(f"生成统计信息时出错: {str(e)}")
        
        # 显示总处理时间
        elapsed_time = time.time() - start_time
        logger.info(f"总处理时间: {elapsed_time:.2f} 秒")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("程序被用户中断")
        sys.exit(1)
    except Exception as e:
        logger.error(f"程序异常终止: {str(e)}", exc_info=True)
        sys.exit(1)
