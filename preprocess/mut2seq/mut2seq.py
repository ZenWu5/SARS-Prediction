import os
import pandas as pd
import re
from multiprocessing import Pool, cpu_count
import logging
from collections import Counter, defaultdict
import sys
from tqdm import tqdm
import numpy as np

# 设置日志记录
def setup_logger(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, 'preprocess.log')
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # 清除已有的处理器
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # 添加文件处理器，使用UTF-8编码
    file_handler = logging.FileHandler(log_path, encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    return logger

# 用于命令行输出
def console_print(message):
    print(message, file=sys.stdout, flush=True)

# 参考序列
UNIPROT_S_PROTEIN = (
    'NITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYFPLQSYGFQPTYGVGYQPYRVVVLSFELLHAPATVCGPKKST'
)

# 处理范围定义
MAX_POS = 201  # 最大处理位置

# 存储序列验证结果
verification_stats = Counter()
target_mismatch_stats = defaultdict(Counter)  # 按target统计不匹配情况
target_barcodes = defaultdict(set)  # 各target包含的barcode
mismatch_barcodes = set()  # 存储原始氨基酸不匹配的条形码
out_of_range_barcodes = set()  # 存储含有超出范围突变的条形码
deletion_barcodes = set()  # 存储含有缺失的条形码

# 第一遍扫描数据集，收集各target的氨基酸信息
def collect_target_reference_data(file_paths):
    console_print("第一步: 按病毒株(target)收集氨基酸参考信息...")
    
    # 对每个target，记录每个位置的氨基酸频率
    target_position_aa = defaultdict(lambda: defaultdict(Counter))
    all_targets = set()
    
    for file_path in file_paths:
        try:
            df = pd.read_csv(file_path)
            
            # 确定target列
            target_col = None
            if 'target' in df.columns:
                target_col = 'target'
            elif 'variant_class' in df.columns:
                target_col = 'variant_class'
            
            if target_col is None:
                console_print(f"警告: 文件 {file_path} 中找不到target或variant_class列，将使用'unknown'作为target")
            
            # 解析aa_substitutions列，按target分组统计
            for mut_aa, row in tqdm(df.iterrows(), total=len(df), desc=f"扫描 {os.path.basename(file_path)}"):
                target = row.get(target_col, 'unknown') if target_col else 'unknown'
                all_targets.add(target)
                
                barcode = row.get('barcode', f"seq_{mut_aa}")
                target_barcodes[target].add(barcode)
                
                aa_sub = row.get('aa_substitutions', '')
                if pd.isna(aa_sub) or aa_sub == "":
                    continue
                
                for mut in aa_sub.split():
                    # 处理标准氨基酸替换格式 (如 A123B)
                    match = re.match(r"([A-Z])(\d+)([A-Z])", mut)
                    if match:
                        orig_aa, pos, mut_aa = match.groups()
                        pos = int(pos)
                        if 1 <= pos <= MAX_POS:
                            target_position_aa[target][pos][orig_aa] += 1
                        continue
                    
                    # 处理氨基酸缺失格式 (如 A123*)
                    match = re.match(r"([A-Z])(\d+)\*", mut)
                    if match:
                        orig_aa, pos = match.groups()
                        pos = int(pos)
                        if 1 <= pos <= MAX_POS:
                            target_position_aa[target][pos][orig_aa] += 1
                        continue
        except Exception as e:
            console_print(f"处理文件 {file_path} 时出错: {str(e)}")
    
    # 构建每个target的参考序列
    target_reference_seqs = {}
    target_replacements = {}
    
    for target in all_targets:
        # 从UniProt序列开始
        target_seq = list(UNIPROT_S_PROTEIN[:MAX_POS])
        replacements = []
        
        # 根据每个位置的最常见氨基酸进行替换
        for pos in range(1, MAX_POS + 1):
            if pos in target_position_aa[target] and target_position_aa[target][pos]:
                # 找出在这个位置出现最多的原始氨基酸
                most_common_aa = target_position_aa[target][pos].most_common(1)[0][0]
                uniprot_aa = UNIPROT_S_PROTEIN[pos-1] if pos <= len(UNIPROT_S_PROTEIN) else 'X'
                
                if most_common_aa != uniprot_aa:
                    target_seq[pos-1] = most_common_aa
                    replacements.append((pos, uniprot_aa, most_common_aa))
        
        target_reference_seqs[target] = ''.join(target_seq)
        target_replacements[target] = replacements
    
    # 记录调整信息
    console_print(f"发现 {len(all_targets)} 个不同的target")
    for target in all_targets:
        n_replacements = len(target_replacements[target])
        n_barcodes = len(target_barcodes[target])
        console_print(f"Target '{target}': {n_barcodes} 条序列, {n_replacements} 个替换位置")
    
    return target_reference_seqs, target_replacements, target_barcodes

# 解析aa_substitutions列为标准格式
def parse_aa_substitutions(row, target_reference_seqs):
    aa_sub = row.get('aa_substitutions', '')
    barcode = row.get('barcode', 'unknown')
    
    # 确定target
    target = None
    if 'target' in row:
        target = row['target']
    elif 'variant_class' in row:
        target = row['variant_class']
    else:
        target = 'unknown'
    
    # 获取对应target的参考序列
    reference = target_reference_seqs.get(target, UNIPROT_S_PROTEIN[:MAX_POS])
    
    if pd.isna(aa_sub) or aa_sub == "":
        return []
    
    mutations = []
    has_deletion = False
    
    for mut in aa_sub.split():
        # 处理标准氨基酸替换格式 (如 A123B)
        match = re.match(r"([A-Z])(\d+)([A-Z])", mut)
        if match:
            orig, pos, new = match.groups()
            pos = int(pos)
            
            # 验证原始氨基酸
            if 1 <= pos <= len(reference):
                ref_aa = reference[pos-1]
                if orig != ref_aa:
                    verification_stats[f"原始氨基酸不匹配: 位置{pos}期望{ref_aa}但得到{orig}"] += 1
                    target_mismatch_stats[target][f"位置{pos}: {ref_aa}->{orig}"] += 1
                    mismatch_barcodes.add(barcode)
                
                # 所有位置都包含在目标序列中
                mutations.append(f"{pos:03d}{new}")
            else:
                verification_stats[f"位置超出调整后序列范围: {pos}"] += 1
                out_of_range_barcodes.add(barcode)
            continue
        
        # 处理氨基酸缺失格式 (如 A123*)
        match = re.match(r"([A-Z])(\d+)\*", mut)
        if match:
            orig, pos = match.groups()
            pos = int(pos)
            
            # 验证原始氨基酸
            if 1 <= pos <= len(reference):
                ref_aa = reference[pos-1]
                if orig != ref_aa:
                    verification_stats[f"原始氨基酸不匹配(缺失): 位置{pos}期望{ref_aa}但得到{orig}"] += 1
                    target_mismatch_stats[target][f"位置{pos}(缺失): {ref_aa}->{orig}"] += 1
                    mismatch_barcodes.add(barcode)
                
                # 标记为缺失，使用'-'表示缺失 (也可以使用其他特殊符号)
                mutations.append(f"{pos:03d}-")
                has_deletion = True
            else:
                verification_stats[f"位置超出调整后序列范围(缺失): {pos}"] += 1
                out_of_range_barcodes.add(barcode)
            continue
            
        # 如果不是以上任何格式
        verification_stats[f"无效突变格式: {mut}"] += 1
    
    if has_deletion:
        deletion_barcodes.add(barcode)
    
    return mutations

# 处理突变列
def process_mutations(df, target_reference_seqs):
    # 使用tqdm显示进度
    tqdm.pandas(desc="解析突变")
    df['label_list'] = df.progress_apply(
        lambda row: parse_aa_substitutions(row, target_reference_seqs), axis=1
    )
    return df

# 生成目标序列
def get_aim_seq(row, target_reference_seqs):
    # 确定target
    target = None
    if 'target' in row:
        target = row['target']
    elif 'variant_class' in row:
        target = row['variant_class']
    else:
        target = 'unknown'
    
    # 获取对应target的参考序列
    reference = target_reference_seqs.get(target, UNIPROT_S_PROTEIN[:MAX_POS])
    modified_seq = list(reference)  # 使用对应target的调整后参考序列
    
    for label in row['label_list']:
        if not label or len(label) != 4:
            continue
            
        try:
            pos = int(label[0:3])  # 原始位置
            index = pos - 1  # 在序列中的索引
            aim_aa = label[3]
            
            if 0 <= index < len(modified_seq):
                # 处理缺失 (用'-'表示)
                if aim_aa == '-':
                    # 对于缺失，可以选择用特定符号替代或者其他处理方式
                    # 这里我们简单地用'-'表示缺失位点
                    modified_seq[index] = '-'
                else:
                    modified_seq[index] = aim_aa  # 替换目标氨基酸
            else:
                verification_stats[f"意外错误: {label} (位置{pos})超出序列范围"] += 1
        except Exception as e:
            verification_stats[f"处理标签 {label} 时出错: {str(e)}"] += 1
    
    return "".join(modified_seq)  # 转换回字符串

if __name__ == '__main__':
    # 确保输出目录存在
    output_dir = 'preprocess/mut2seq/output'
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置日志
    logger = setup_logger(output_dir)
    
    # 文件路径列表
    file_paths = [
        'data/bc_binding_omicron.csv',
        'data/bc_binding_variants.csv'
        # 这里可添加更多文件路径
    ]

    # 收集各target的氨基酸信息并构建参考序列
    target_reference_seqs, target_replacements, target_barcodes = collect_target_reference_data(file_paths)
    
    # 记录各target的参考序列
    references_output_path = os.path.join(output_dir, 'target_references.fasta')
    with open(references_output_path, 'w', encoding='utf-8') as f:
        for target, seq in target_reference_seqs.items():
            f.write(f">{target}\n")
            f.write(f"{seq}\n")
    
    console_print(f"各target的参考序列已保存至 {references_output_path}")
    
    # 记录详细的替换信息
    replacements_output_path = os.path.join(output_dir, 'target_replacements.txt')
    with open(replacements_output_path, 'w', encoding='utf-8') as f:
        for target, replacements in target_replacements.items():
            f.write(f"Target: {target}, 序列数: {len(target_barcodes[target])}, 替换位置数: {len(replacements)}\n")
            for pos, old_aa, new_aa in replacements:
                f.write(f"  位置 {pos}: {old_aa} -> {new_aa}\n")
            f.write("\n")
    
    console_print(f"各target的替换位置信息已保存至 {replacements_output_path}")
    
    # 记录日志信息
    logger.info(f"发现 {len(target_reference_seqs)} 个不同的target")
    for target in target_reference_seqs:
        logger.info(f"Target '{target}': {len(target_barcodes[target])} 条序列, {len(target_replacements[target])} 个替换位置")
    
    console_print(f"第二步: 使用各target的参考序列处理数据...")
    console_print(f"处理范围: 1-{MAX_POS}")

    # 初始化一个列表用于存储处理后的数据
    sum_data = []
    total_sequences = 0
    
    # 处理每个数据集
    for file_idx, file_path in enumerate(file_paths, 1):
        try:
            console_print(f"[{file_idx}/{len(file_paths)}] 处理文件: {file_path}")
            logger.info(f"开始处理文件: {file_path}")
            
            df = pd.read_csv(file_path)
            file_sequences = len(df)
            total_sequences += file_sequences
            console_print(f"  - 读取 {file_sequences} 行数据")
            
            # 确保有barcode列，如果没有则添加一个
            if 'barcode' not in df.columns:
                df['barcode'] = [f"seq_{i}" for i in range(1, len(df) + 1)]
                logger.info(f"文件没有barcode列，已自动生成")
            
            # 处理突变
            df = process_mutations(df, target_reference_seqs)
            console_print(f"  - 突变解析完成")

            # 生成目标序列
            console_print(f"  - 生成目标序列...")
            tqdm.pandas(desc="生成序列")
            df['aim_seq'] = df.progress_apply(
                lambda row: get_aim_seq(row, target_reference_seqs), axis=1
            )

            # 保留指定列
            keep_columns = ['library','variant_class', 'target', 'n_aa_substitutions', 'TiteSeq_avgcount', 
                            'log10Ka', 'aim_seq', 'barcode', 'aa_substitutions']
            # 只保留存在的列
            avail_columns = [col for col in keep_columns if col in df.columns]
            if 'aim_seq' not in avail_columns:
                avail_columns.append('aim_seq')  # 确保aim_seq列一定被保留
            
            # 移除重复列名
            avail_columns = list(dict.fromkeys(avail_columns))
            
            sum_data.append(df[avail_columns])
            logger.info(f"文件 {file_path} 处理完成，共 {len(df)} 行数据")
            console_print(f"  - 完成！保留列: {', '.join(avail_columns)}")
        except Exception as e:
            verification_stats[f"处理文件 {file_path} 时出错: {str(e)}"] += 1
            logger.error(f"处理文件 {file_path} 时出错: {e}", exc_info=True)
            console_print(f"  - 错误: {str(e)}")

    # 合并所有数据集
    if sum_data:
        console_print("合并所有数据集...")
        final_df = pd.concat(sum_data, ignore_index=True)

        # 输出为 CSV
        output_path = os.path.join(output_dir, 'merged_variant_output.csv')
        final_df.to_csv(output_path, index=False, encoding='utf-8-sig')

        # # 输出为 NPY
        # output_path = os.path.join(output_dir, 'merged_variant_output.npy')
        # final_df.to_numpy().dump(output_path)
        # console_print(f"合并完成！数据已保存至 {output_path}")

        # 输出为NPZ
        # output_path = os.path.join(output_dir, 'merged_variant_output.npz')
        # np.savez_compressed(
        #     output_path,
        #     data=final_df.to_numpy(),
        #     columns=final_df.columns.to_list()
        # )
        # console_print(f"合并完成！数据已保存至 {output_path}")
        
        # 记录按target分类的不匹配统计
        target_stats_path = os.path.join(output_dir, 'target_mismatch_stats.txt')
        with open(target_stats_path, 'w', encoding='utf-8') as f:
            for target in sorted(target_mismatch_stats.keys()):
                f.write(f"Target: {target}\n")
                for mismatch, count in target_mismatch_stats[target].most_common():
                    f.write(f"  {mismatch}: {count} 次\n")
                f.write(f"  总不匹配数: {sum(target_mismatch_stats[target].values())}\n")
                f.write(f"  总序列数: {len(target_barcodes[target])}\n")
                mismatch_percent = 100 * sum(target_mismatch_stats[target].values()) / len(target_barcodes[target]) if target_barcodes[target] else 0
                f.write(f"  不匹配序列比例: {mismatch_percent:.2f}%\n\n")
        
        # 计算统计信息
        out_of_range_count = len(out_of_range_barcodes)
        mismatch_count = len(mismatch_barcodes)
        deletion_count = len(deletion_barcodes)
        problematic_count = len(out_of_range_barcodes.union(mismatch_barcodes))
        
        out_of_range_percentage = (out_of_range_count / total_sequences) * 100 if total_sequences > 0 else 0
        mismatch_percentage = (mismatch_count / total_sequences) * 100 if total_sequences > 0 else 0
        deletion_percentage = (deletion_count / total_sequences) * 100 if total_sequences > 0 else 0
        problematic_percentage = (problematic_count / total_sequences) * 100 if total_sequences > 0 else 0
        
        # 输出统计信息
        summary_message = [
            f"处理完成! 数据已保存至 {output_path}",
            f"总序列数: {total_sequences}",
            f"各target的参考序列已保存至 {references_output_path}",
            f"各target的替换信息已保存至 {replacements_output_path}",
            f"各target的不匹配统计已保存至 {target_stats_path}",
            f"处理范围: 1-{MAX_POS}",
            f"含有超出范围突变的序列: {out_of_range_count} 条，占总数的 {out_of_range_percentage:.2f}%",
            f"原始氨基酸不匹配的序列: {mismatch_count} 条，占总数的 {mismatch_percentage:.2f}%",
            f"含有氨基酸缺失的序列: {deletion_count} 条，占总数的 {deletion_percentage:.2f}%",
            f"存在任何问题的序列: {problematic_count} 条，占总数的 {problematic_percentage:.2f}%"
        ]
        
        # 记录到日志
        logger.info(f"所有数据已处理并保存至 {output_path}，共 {len(final_df)} 行数据")
        logger.info(f"各target的参考序列已保存至 {references_output_path}")
        logger.info(f"各target的替换信息已保存至 {replacements_output_path}")
        logger.info(f"各target的不匹配统计已保存至 {target_stats_path}")
        logger.info("\n".join(summary_message[5:]))
        logger.info("错误统计:")
        for error, count in verification_stats.most_common():
            logger.info(f"  {error}: {count} 次")
        
        # 输出到命令行
        console_print("\n" + "\n".join(summary_message))
        console_print(f"错误类型总数: {len(verification_stats)}")
        if verification_stats:
            console_print(f"最常见错误: {verification_stats.most_common(1)[0][0]}, 出现 {verification_stats.most_common(1)[0][1]} 次")
            # 输出前5个最常见错误
            console_print("前5个最常见错误:")
            for error, count in verification_stats.most_common(5):
                console_print(f"  - {error}: {count} 次")
        console_print(f"详细日志已保存至 {os.path.join(output_dir, 'preprocess.log')}")
    else:
        logger.warning("未找到可处理的数据。")
        console_print("错误: 未找到可处理的数据。")
