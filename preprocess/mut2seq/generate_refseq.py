import re
from collections import Counter
import pandas as pd

def get_aa_sub_list(df, target):
    """
    从数据框中提取突变信息（A123B）并返回一个列表。
    
    参数:
    - df: 包含突变信息的数据框
    
    返回:
    - 包含突变信息的字符串列表（如 ['A123B', 'Y456F']）
    """
    # 提取突变信息列
    aa_sub_list = df['aa_substitutions'].tolist()
    
    # 过滤掉空值
    aa_sub_list = [x for x in aa_sub_list if pd.notna(x)]
    
    return aa_sub_list

def count_unique_positions(aa_sub_list):
    """
    统计突变字符串列表中所有唯一位置的总数。
    
    参数:
    - aa_sub_list: 一个包含突变信息的字符串列表（如 ['A123B', 'Y456F']）
    
    返回:
    - 唯一位置的总数（整数），位置的最大数和最小数
    """
    positions = set()  # 使用集合存储唯一位置
    for mutation in aa_sub_list:
        match = re.match(r"[A-Z](\d+)[A-Z]", mutation)
        if match:
            position = int(match.group(1))
            positions.add(position)  # 添加位置到集合中
    return len(positions), max(positions), min(positions)  # 返回集合中位置的数量


def reconstruct_original_sequence(aa_sub_list, max_pos):
    """
    根据突变信息（A123B）中的原始序列信息反推出突变前的序列。
    
    参数:
    - aa_sub_list: 一个包含突变信息的字符串列表（如 ['A123B', 'Y456F']）
    
    返回:
    - 由原始氨基酸推测构建的序列（字符串）
    """
    # 初始化一个字典用于存储重建的序列
    reconstructed_seq = {}
    
    # 解析每个突变，提取原始氨基酸和位置
    for mutation in aa_sub_list:
        # 使用正则表达式提取信息：第一个字母是原始氨基酸，中间的数字是位置
        match = re.match(r'([A-Z])(\d+)[A-Z]', mutation)
        if match:
            original_aa, position = match.groups()
            position = int(position)
            
            if 1 <= position < max_pos:
                # 将原始氨基酸添加到对应位置
                if position in reconstructed_seq:
                    reconstructed_seq[position].append(original_aa)
                else:
                    reconstructed_seq[position] = [original_aa]
    
    # 构建最终序列
    final_sequence = ['-'] * (max_pos - 1)  # 初始化所有位置为'-'
    
    # 填充已知位置的氨基酸
    for pos, aa_list in reconstructed_seq.items():
        # 如果有多个值，选择出现次数最多的
        most_common_aa = Counter(aa_list).most_common(1)[0][0]
        final_sequence[pos-1] = most_common_aa  # 位置从1开始，索引从0开始
    
    return ''.join(final_sequence)


def main():
    import_dir = r'data\bc_binding_omicron.csv'
    output_fasta = r'preprocess\mut2seq\output\re_seq_omicron.fasta'

    # 加载数据
    df = pd.read_csv(import_dir)

    # 检查数据是否包含目标列
    if 'target' not in df.columns:
        raise ValueError("数据框中缺少'target'列")
    if 'aa_substitutions' not in df.columns:
        raise ValueError("数据框中缺少'aa_substitutions'列")
    
    # 初始化字典用于存储结果
    result_dict = {}

    # 获取target列
    targetlist = df['target'].unique()
    
    # 遍历每个target
    for target in targetlist:
        print(f"处理目标: {target}")
        try:
            # 获取突变信息列表
            aa_sub_list = get_aa_sub_list(df, target)
            
            # 统计唯一位置的总数
            unique_positions_count, max_pos, min_pos = count_unique_positions(aa_sub_list)
            print(f"病毒株 {target} 唯一位置的总数: {unique_positions_count}, 最大位置: {max_pos}, 最小位置: {min_pos}")
            
            # 重建原始序列
            original_sequence = reconstruct_original_sequence(aa_sub_list, max_pos)
            
            # 将 target 和重建的序列存入字典
            result_dict[target] = original_sequence

        except FileNotFoundError:
            print(f"文件未找到: {import_dir}")
        except ValueError as e:
            print(f"数据处理错误: {e}")
        except Exception as e:
            print(f"发生未知错误: {e}")
    
    # 导出结果为 FASTA 格式
    with open(output_fasta, 'w') as fasta_file:
        for target, sequence in result_dict.items():
            fasta_file.write(f">{target}\n")
            fasta_file.write(f"{sequence}\n")
    
    print(f"FASTA 文件已保存至: {output_fasta}")

if __name__ == "__main__":
    main()