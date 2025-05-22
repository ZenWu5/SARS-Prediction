import os
import numpy as np
import h5py
import torch
from tqdm import tqdm
import argparse
import json
import gc
import shutil
import tempfile
import psutil
from pathlib import Path
import sys

def get_available_memory():
    """获取可用内存(GB)"""
    return psutil.virtual_memory().available / (1024 * 1024 * 1024)

def get_available_disk_space(path):
    """获取指定路径所在磁盘的可用空间(GB)"""
    return shutil.disk_usage(path).free / (1024 * 1024 * 1024)

def compute_stats(h5_path, sample_size=1000):
    """一次性计算embedding和target的统计信息，减少文件打开次数"""
    with h5py.File(h5_path, 'r') as f:
        total = int(f.attrs['total_samples'])
        # 计算目标值统计
        targets = f['mean_log10Ka'][:].astype(np.float32)
        target_mean = np.mean(targets)
        target_std = np.std(targets) + 1e-8
        
        # 采样计算embedding统计
        indices = np.random.choice(total, min(sample_size, total), replace=False)
        all_embs = []
        for idx in tqdm(indices, desc="采样embedding"):
            all_embs.append(f['embeddings'][f'emb_{idx}'][:])
        all_embs = np.vstack(all_embs)
        emb_mean = np.mean(all_embs, axis=0)
        emb_std = np.std(all_embs, axis=0) + 1e-8
        
        # 清理内存
        del all_embs
        gc.collect()
    
    return emb_mean, emb_std, target_mean, target_std

def safe_save_torch(data, filepath):
    """安全地保存PyTorch数据，使用临时文件避免写入错误"""
    try:
        # 使用临时文件先保存
        temp_filepath = filepath + ".tmp"
        torch.save(data, temp_filepath)
        
        # 如果成功保存，重命名为最终文件名
        if os.path.exists(temp_filepath):
            if os.path.exists(filepath):
                os.remove(filepath)
            shutil.move(temp_filepath, filepath)
            return True
        return False
    except Exception as e:
        print(f"保存文件 {filepath} 时出错: {e}")
        if os.path.exists(temp_filepath):
            os.remove(temp_filepath)
        return False

def save_batch(batch_data, output_path, filetype="pt"):
    """保存单个批次数据"""
    features, masks, targets, diff_counts = batch_data
    
    try:
        if filetype == "pt":
            data_dict = {
                'features': torch.tensor(features),
                'masks': torch.tensor(masks),
                'targets': torch.tensor(targets),
                'diff_counts': torch.tensor(diff_counts)
            }
            filepath = f"{output_path}.{filetype}"
            success = safe_save_torch(data_dict, filepath)
            
            if not success:
                # 尝试使用NPZ格式保存
                print(f"PyTorch保存失败，尝试使用NPZ格式保存 {output_path}")
                np.savez_compressed(f"{output_path}.npz", 
                    features=features, masks=masks, 
                    targets=targets, diff_counts=diff_counts)
        else:
            np.savez_compressed(f"{output_path}.{filetype}", 
                features=features, masks=masks, 
                targets=targets, diff_counts=diff_counts)
    except Exception as e:
        print(f"保存批次数据时出错: {e}")
        # 尝试使用纯NumPy保存
        try:
            print("尝试使用纯NumPy格式保存...")
            np.save(f"{output_path}_features.npy", features)
            np.save(f"{output_path}_masks.npy", masks)
            np.save(f"{output_path}_targets.npy", targets)
            np.save(f"{output_path}_diff_counts.npy", diff_counts)
        except Exception as e2:
            print(f"备用保存方法也失败: {e2}")
    
    # 清理内存
    del features, masks, targets, diff_counts
    gc.collect()

def process_batch(batch_idx, h5_path, start, end, max_len, emb_mean, emb_std, target_mean, target_std):
    """处理单个批次，可用于并行处理"""
    try:
        with h5py.File(h5_path, 'r') as f:
            N = end - start
            emb_dim = int(f.attrs['embedding_dim'])
            features = np.zeros((N, max_len, emb_dim), dtype=np.float32)
            masks = np.zeros((N, max_len), dtype=bool)
            t_batch = np.zeros((N,), dtype=np.float32)
            d_batch = np.zeros((N,), dtype=np.int16)
            
            for i, idx in enumerate(range(start, end)):
                emb = f['embeddings'][f'emb_{idx}'][:].astype(np.float32)
                emb = (emb - emb_mean) / emb_std
                L = emb.shape[0]
                
                # 更高效的填充方式
                features[i, :min(L, max_len)] = emb[:min(L, max_len)]
                masks[i, :min(L, max_len)] = True
                
                t_batch[i] = (f['mean_log10Ka'][idx] - target_mean) / target_std
                d_batch[i] = f['diff_count'][idx]
        
        return batch_idx, (features, masks, t_batch, d_batch)
    except Exception as e:
        print(f"处理批次 {batch_idx} 时出错: {e}")
        raise

def stream_merge_to_h5(batch_files, output_file):
    """采用HDF5流式合并文件，解决内存限制问题"""
    print(f"开始HDF5流式合并，处理 {len(batch_files)} 个批次文件...")
    
    # 获取数据维度和总样本数
    total_samples = 0
    embedding_dim = None
    max_seq_len = None
    
    # 先查看第一个文件确定维度信息
    for file in batch_files:
        try:
            if file.endswith('.pt'):
                data = torch.load(file)
                sample_count = data['features'].shape[0]
                if embedding_dim is None:
                    if isinstance(data['features'], torch.Tensor):
                        embedding_dim = data['features'].shape[2]
                        max_seq_len = data['features'].shape[1]
                    else:
                        embedding_dim = data['features'].shape[2]
                        max_seq_len = data['features'].shape[1]
            else:  # npz文件
                data = np.load(file)
                sample_count = data['features'].shape[0]
                if embedding_dim is None:
                    embedding_dim = data['features'].shape[2]
                    max_seq_len = data['features'].shape[1]
            
            total_samples += sample_count
            del data
            gc.collect()
        except Exception as e:
            print(f"读取文件 {file} 时出错: {e}")
            continue
    
    if embedding_dim is None or max_seq_len is None:
        print("无法确定数据维度，合并失败")
        return False
    
    print(f"总样本数: {total_samples}, 序列最大长度: {max_seq_len}, 嵌入维度: {embedding_dim}")
    
    # 创建输出HDF5文件
    h5_output = output_file.replace('.pt', '.h5').replace('.npz', '.h5')
    print(f"创建输出文件: {h5_output}")
    
    try:
        with h5py.File(h5_output, 'w') as f:
            # 设置属性
            f.attrs['total_samples'] = total_samples
            f.attrs['max_seq_len'] = max_seq_len
            f.attrs['embedding_dim'] = embedding_dim
            
            # 创建数据集
            features_ds = f.create_dataset('features', 
                                        shape=(total_samples, max_seq_len, embedding_dim),
                                        dtype=np.float32,
                                        chunks=(1, max_seq_len, embedding_dim),
                                        compression='gzip',
                                        compression_opts=4)
            
            masks_ds = f.create_dataset('masks',
                                      shape=(total_samples, max_seq_len),
                                      dtype=bool,
                                      chunks=(100, max_seq_len),
                                      compression='gzip',
                                      compression_opts=4)
            
            targets_ds = f.create_dataset('targets',
                                       shape=(total_samples,),
                                       dtype=np.float32)
            
            diff_counts_ds = f.create_dataset('diff_counts',
                                           shape=(total_samples,),
                                           dtype=np.int16)
            
            # 流式写入数据
            start_idx = 0
            for i, file in enumerate(tqdm(batch_files, desc="合并批次文件")):
                try:
                    # 加载批次文件
                    if file.endswith('.pt'):
                        try:
                            data = torch.load(file)
                            batch_size = data['features'].shape[0]
                            
                            # 转换为NumPy
                            if isinstance(data['features'], torch.Tensor):
                                features = data['features'].numpy()
                                masks = data['masks'].numpy()
                                targets = data['targets'].numpy()
                                diff_counts = data['diff_counts'].numpy()
                            else:
                                features = data['features']
                                masks = data['masks']
                                targets = data['targets']
                                diff_counts = data['diff_counts']
                        except Exception as e:
                            print(f"加载PyTorch文件 {file} 失败: {e}")
                            
                            # 尝试加载对应的NPZ文件
                            npz_file = file.replace('.pt', '.npz')
                            if os.path.exists(npz_file):
                                data = np.load(npz_file)
                                batch_size = data['features'].shape[0]
                                features = data['features']
                                masks = data['masks']
                                targets = data['targets']
                                diff_counts = data['diff_counts']
                            else:
                                # 尝试加载分散的NPY文件
                                base_name = file.replace('.pt', '')
                                features_file = f"{base_name}_features.npy"
                                
                                if os.path.exists(features_file):
                                    features = np.load(features_file)
                                    masks = np.load(f"{base_name}_masks.npy")
                                    targets = np.load(f"{base_name}_targets.npy")
                                    diff_counts = np.load(f"{base_name}_diff_counts.npy")
                                    batch_size = features.shape[0]
                                else:
                                    print(f"找不到任何可用的备份文件，跳过 {file}")
                                    continue
                    else:  # npz文件
                        data = np.load(file)
                        batch_size = data['features'].shape[0]
                        features = data['features']
                        masks = data['masks']
                        targets = data['targets']
                        diff_counts = data['diff_counts']
                    
                    # 写入HDF5文件
                    end_idx = start_idx + batch_size
                    features_ds[start_idx:end_idx] = features
                    masks_ds[start_idx:end_idx] = masks
                    targets_ds[start_idx:end_idx] = targets
                    diff_counts_ds[start_idx:end_idx] = diff_counts
                    
                    start_idx = end_idx
                    
                    # 清理内存
                    del data, features, masks, targets, diff_counts
                    gc.collect()
                    
                except Exception as e:
                    print(f"处理文件 {file} 时出错: {e}")
                    continue
    except Exception as e:
        print(f"创建或写入HDF5文件时出错: {e}")
        return False
    
    # 尝试转换为NPZ格式（如果需要）
    if output_file.endswith('.pt') or output_file.endswith('.npz'):
        print(f"数据已合并到HDF5文件: {h5_output}")
        print("注意: 由于数据量大，输出格式已改为HDF5。请使用h5py库加载数据。")
        
        # 创建加载示例文件
        example_file = h5_output.replace('.h5', '_load_example.py')
        with open(example_file, 'w') as f:
            f.write(format(os.path.basename(h5_output)))
        print(f"已创建加载示例代码: {example_file}")
    
    return True

def stream_merge_to_pt(batch_files, output_file, max_chunk=5000):
    """
    流式合并到PyTorch文件，通过分块处理减少内存使用
    max_chunk: 每次处理的最大样本数
    """
    print(f"开始流式合并到PyTorch文件，处理 {len(batch_files)} 个批次文件...")
    
    # 获取数据维度和总样本数
    total_samples = 0
    embedding_dim = None
    max_seq_len = None
    
    # 先查看所有文件确定总样本数和维度信息
    for file in tqdm(batch_files, desc="扫描文件统计信息"):
        try:
            if file.endswith('.pt'):
                data = torch.load(file)
                sample_count = data['features'].shape[0]
                if embedding_dim is None:
                    if isinstance(data['features'], torch.Tensor):
                        embedding_dim = data['features'].shape[2]
                        max_seq_len = data['features'].shape[1]
                    else:
                        embedding_dim = data['features'].shape[2]
                        max_seq_len = data['features'].shape[1]
            else:  # npz文件
                data = np.load(file)
                sample_count = data['features'].shape[0]
                if embedding_dim is None:
                    embedding_dim = data['features'].shape[2]
                    max_seq_len = data['features'].shape[1]
            
            total_samples += sample_count
            del data
            gc.collect()
        except Exception as e:
            print(f"读取文件 {file} 时出错: {e}")
            continue
    
    if embedding_dim is None or max_seq_len is None:
        print("无法确定数据维度，合并失败")
        return False
    
    print(f"总样本数: {total_samples}, 序列最大长度: {max_seq_len}, 嵌入维度: {embedding_dim}")
    
    # 计算需要的内存大小
    chunk_size_gb = (max_chunk * max_seq_len * embedding_dim * 4) / (1024**3)  # 4字节/float32
    print(f"每次处理 {max_chunk} 个样本，估计内存需求: {chunk_size_gb:.2f} GB")
    
    # 创建存储结果的字典
    merged_data = {
        'features': torch.zeros((total_samples, max_seq_len, embedding_dim), dtype=torch.float32),
        'masks': torch.zeros((total_samples, max_seq_len), dtype=torch.bool),
        'targets': torch.zeros((total_samples,), dtype=torch.float32),
        'diff_counts': torch.zeros((total_samples,), dtype=torch.int16)
    }
    
    try:
        # 分批次填充数据
        start_idx = 0
        
        for file_batch_idx, file in enumerate(tqdm(batch_files, desc="合并批次文件")):
            try:
                # 加载批次文件
                if file.endswith('.pt'):
                    try:
                        data = torch.load(file)
                        batch_size = data['features'].shape[0]
                        
                        # 确保数据是PyTorch张量
                        if not isinstance(data['features'], torch.Tensor):
                            features = torch.tensor(data['features'], dtype=torch.float32)
                            masks = torch.tensor(data['masks'], dtype=torch.bool)
                            targets = torch.tensor(data['targets'], dtype=torch.float32)
                            diff_counts = torch.tensor(data['diff_counts'], dtype=torch.int16)
                        else:
                            features = data['features']
                            masks = data['masks']
                            targets = data['targets']
                            diff_counts = data['diff_counts']
                    except Exception as e:
                        print(f"加载PyTorch文件 {file} 失败: {e}")
                        
                        # 尝试加载对应的NPZ文件
                        npz_file = file.replace('.pt', '.npz')
                        if os.path.exists(npz_file):
                            data = np.load(npz_file)
                            batch_size = data['features'].shape[0]
                            features = torch.tensor(data['features'], dtype=torch.float32)
                            masks = torch.tensor(data['masks'], dtype=torch.bool)
                            targets = torch.tensor(data['targets'], dtype=torch.float32)
                            diff_counts = torch.tensor(data['diff_counts'], dtype=torch.int16)
                        else:
                            # 尝试加载分散的NPY文件
                            base_name = file.replace('.pt', '')
                            features_file = f"{base_name}_features.npy"
                            
                            if os.path.exists(features_file):
                                features_np = np.load(features_file)
                                masks_np = np.load(f"{base_name}_masks.npy")
                                targets_np = np.load(f"{base_name}_targets.npy")
                                diff_counts_np = np.load(f"{base_name}_diff_counts.npy")
                                
                                batch_size = features_np.shape[0]
                                features = torch.tensor(features_np, dtype=torch.float32)
                                masks = torch.tensor(masks_np, dtype=torch.bool)
                                targets = torch.tensor(targets_np, dtype=torch.float32)
                                diff_counts = torch.tensor(diff_counts_np, dtype=torch.int16)
                                
                                del features_np, masks_np, targets_np, diff_counts_np
                            else:
                                print(f"找不到任何可用的备份文件，跳过 {file}")
                                continue
                else:  # npz文件
                    data = np.load(file)
                    batch_size = data['features'].shape[0]
                    features = torch.tensor(data['features'], dtype=torch.float32)
                    masks = torch.tensor(data['masks'], dtype=torch.bool)
                    targets = torch.tensor(data['targets'], dtype=torch.float32)
                    diff_counts = torch.tensor(data['diff_counts'], dtype=torch.int16)
                
                # 分块填充数据
                end_idx = start_idx + batch_size
                
                # 分块填充大型数据
                if batch_size > max_chunk:
                    for chunk_start in range(0, batch_size, max_chunk):
                        chunk_end = min(chunk_start + max_chunk, batch_size)
                        data_start = start_idx + chunk_start
                        data_end = start_idx + chunk_end
                        
                        # 复制数据块
                        merged_data['features'][data_start:data_end] = features[chunk_start:chunk_end]
                        merged_data['masks'][data_start:data_end] = masks[chunk_start:chunk_end]
                        
                        # 强制内存回收
                        gc.collect()
                    
                    # 复制剩余较小的数组
                    merged_data['targets'][start_idx:end_idx] = targets
                    merged_data['diff_counts'][start_idx:end_idx] = diff_counts
                else:
                    # 对于小批次，直接复制
                    merged_data['features'][start_idx:end_idx] = features
                    merged_data['masks'][start_idx:end_idx] = masks
                    merged_data['targets'][start_idx:end_idx] = targets
                    merged_data['diff_counts'][start_idx:end_idx] = diff_counts
                
                start_idx = end_idx
                
                # 每处理10批次，保存一次中间结果
                if (file_batch_idx + 1) % 10 == 0 or file_batch_idx == len(batch_files) - 1:
                    print(f"保存中间结果 ({start_idx}/{total_samples} 样本)...")
                    torch.save(merged_data, output_file + ".tmp")
                
                # 清理内存
                del features, masks, targets, diff_counts
                if 'data' in locals():
                    del data
                gc.collect()
                
            except Exception as e:
                print(f"处理文件 {file} 时出错: {e}")
                continue
                
        # 保存最终结果
        print(f"保存最终结果到 {output_file}...")
        torch.save(merged_data, output_file)
        
        # 删除临时文件
        if os.path.exists(output_file + ".tmp"):
            os.remove(output_file + ".tmp")
            
        print(f"合并完成! 已保存到 {output_file}")
        return True
        
    except Exception as e:
        print(f"合并过程中出错: {e}")
        
        # 检查是否有临时文件
        if os.path.exists(output_file + ".tmp"):
            print(f"检测到中间结果，尝试恢复...")
            try:
                # 尝试重命名中间结果为最终结果
                shutil.move(output_file + ".tmp", output_file)
                print(f"已恢复部分结果到 {output_file}")
                return True
            except Exception as e2:
                print(f"恢复失败: {e2}")
        
        return False

def main(h5_path, output_prefix, max_seq_len=None, batch_size=1000, save_npz=False, 
         max_workers=4, keep_temp=False, batch_dir=None, force_h5=False, chunk_size=5000):
    # 获取初始信息
    filetype = "npz" if save_npz else "pt"
    h5_path = h5_path if h5_path.endswith('.h5') else h5_path + '.h5'
    
    # 获取初始内存和磁盘状态
    initial_mem = get_available_memory()
    print(f"初始可用内存: {initial_mem:.2f} GB")
    
    # 创建临时目录存放批次文件
    if batch_dir is None:
        # 创建在与输出目录相同的位置，避免跨磁盘操作
        output_dir = os.path.dirname(os.path.abspath(output_prefix))
        temp_dir = tempfile.mkdtemp(prefix="protein_batches_", dir=output_dir)
    else:
        # 使用指定目录
        os.makedirs(batch_dir, exist_ok=True)
        temp_dir = batch_dir
    
    print(f"使用批次目录: {temp_dir}")
    
    # 检查磁盘空间
    available_disk = get_available_disk_space(temp_dir)
    print(f"可用磁盘空间: {available_disk:.2f} GB")
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(os.path.abspath(output_prefix)), exist_ok=True)

    # 获取基本信息
    try:
        with h5py.File(h5_path, 'r') as f:
            total = int(f.attrs['total_samples'])
            emb_dim = int(f.attrs['embedding_dim'])
            
            # 如果没有指定最大长度，计算最大长度
            if max_seq_len is None:
                lengths = []
                sample_size = min(1000, total)
                for i in tqdm(range(sample_size), desc="计算序列长度"):
                    lengths.append(f['embeddings'][f'emb_{i}'].shape[0])
                if sample_size < total:
                    print(f"警告：只采样了前{sample_size}条数据计算长度，完整扫描可能会更准确")
                max_len = max(lengths)
            else:
                max_len = max_seq_len

        print(f"数据总数: {total}, 最大长度: {max_len}, 维度: {emb_dim}")
        
        # 一次性计算统计信息
        emb_mean, emb_std, target_mean, target_std = compute_stats(h5_path)

        num_batches = (total + batch_size - 1) // batch_size
        print(f"将分为 {num_batches} 个批次，每批最多 {batch_size} 条")
    except Exception as e:
        print(f"读取H5文件时出错: {e}")
        sys.exit(1)

    # 处理批次
    batch_files = []
    already_processed = []
    
    # 检查是否已经有处理好的批次文件
    for idx in range(num_batches):
        batch_path = os.path.join(temp_dir, f"batch_{idx}.{filetype}")
        if os.path.exists(batch_path):
            print(f"发现已处理的批次 {idx}")
            batch_files.append(batch_path)
            already_processed.append(idx)
        
        # 检查备用文件格式
        if filetype == "pt":
            npz_path = os.path.join(temp_dir, f"batch_{idx}.npz")
            if os.path.exists(npz_path) and idx not in already_processed:
                print(f"发现已处理的备用批次 {idx} (NPZ格式)")
                batch_files.append(npz_path)
                already_processed.append(idx)
    
    # 处理剩余批次
    from concurrent.futures import ThreadPoolExecutor, as_completed
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有未处理的任务
        futures = []
        for b in range(num_batches):
            if b in already_processed:
                continue
                
            start, end = b * batch_size, min((b + 1) * batch_size, total)
            futures.append(
                executor.submit(
                    process_batch, b, h5_path, start, end, max_len, 
                    emb_mean, emb_std, target_mean, target_std
                )
            )
        
        # 处理结果并保存
        for future in tqdm(as_completed(futures), total=len(futures), desc="处理批次"):
            try:
                batch_idx, batch_data = future.result()
                batch_path = os.path.join(temp_dir, f"batch_{batch_idx}")
                save_batch(batch_data, batch_path, filetype)
                
                # 检查哪种格式的文件成功保存了
                if os.path.exists(f"{batch_path}.{filetype}"):
                    batch_files.append(f"{batch_path}.{filetype}")
                elif os.path.exists(f"{batch_path}.npz"):
                    batch_files.append(f"{batch_path}.npz")
                else:
                    # 检查是否有分散的NPY文件
                    if os.path.exists(f"{batch_path}_features.npy"):
                        print(f"注意: 批次 {batch_idx} 以分散NPY格式保存")
                        # 我们暂时不支持合并分散的NPY文件，但它们已经保存了
            except Exception as e:
                print(f"处理批次时出错: {e}")
                # 继续处理其他批次
    
    # 排序批次文件以确保顺序
    batch_files.sort(key=lambda x: int(Path(x).stem.split("_")[1]))
    
    # 合并所有批次
    merged_path = f"{output_prefix}_merged.{filetype}"
    
    # 估计合并内存需求
    approx_size_gb = (total * max_len * emb_dim * 4) / (1024**3)  # 4字节/float32
    print(f"估计数据集大小: {approx_size_gb:.2f} GB")
    
    # 根据数据大小和用户选择决定合并方法
    if force_h5 or approx_size_gb > initial_mem * 0.7:  # 如果超过可用内存的70%
        print("使用HDF5流式合并方法...")
        merge_success = stream_merge_to_h5(batch_files, merged_path)
    else:
        if filetype == "pt":
            print("使用PyTorch流式合并方法...")
            merge_success = stream_merge_to_pt(batch_files, merged_path, max_chunk=chunk_size)
        else:
            print("使用HDF5流式合并方法...")
            merge_success = stream_merge_to_h5(batch_files, merged_path)
    
    if merge_success:
        print("合并成功!")
    else:
        print("合并失败，但批次文件可能仍然可用")
    
    # 保存归一化统计信息
    stats = {
        'embedding_mean': emb_mean.tolist(),
        'embedding_std': emb_std.tolist(),
        'target_mean': float(target_mean),
        'target_std': float(target_std),
        'max_seq_len': int(max_len)
    }
    
    with open(f"{output_prefix}_norm_stats.json", 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)
    
    # 清理临时文件
    if not keep_temp and batch_dir is None:
        print(f"清理临时目录: {temp_dir}")
        try:
            shutil.rmtree(temp_dir)
        except Exception as e:
            print(f"清理临时文件时出错: {e}")
    else:
        print(f"保留批次文件目录: {temp_dir}")
    
    # 最终内存报告
    final_mem = get_available_memory()
    print(f"最终可用内存: {final_mem:.2f} GB (变化: {final_mem-initial_mem:.2f} GB)")
    
    # 根据实际使用的输出文件格式，显示不同的完成信息
    if os.path.exists(merged_path):
        output_file = merged_path
    else:
        h5_output = merged_path.replace('.pt', '.h5').replace('.npz', '.h5')
        if os.path.exists(h5_output):
            output_file = h5_output
        else:
            output_file = "未能创建合并文件，但批次文件可能仍然可用"
    
    print(f"已完成，结果保存在: \n{output_file}\n{output_prefix}_norm_stats.json")

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="将HDF5蛋白质数据批处理并合并为.pt或.npz")
    parser.add_argument("--input", type=str, default=r'preprocess\ESM\output\sampled_output_esm_embeddings.h5', help="HDF5文件路径")
    parser.add_argument("--output", type=str, default=r'preprocess\ESM\output\scaled\sampled_output_esm_embeddings_scaled', help="输出前缀，不含扩展名")
    parser.add_argument("--max_seq_len", type=int, default=None, help="最大序列长度（可选）")
    parser.add_argument("--batch_size", type=int, default=2000, help="每批大小")
    parser.add_argument("--save_npz", action='store_true', help="保存为npz格式（默认.pt）")
    parser.add_argument("--max_workers", type=int, default=4, help="并行处理的最大工作线程数")
    parser.add_argument("--keep_temp", action='store_true', help="保留临时批次文件")
    parser.add_argument("--batch_dir", type=str, default=r'preprocess\ESM\output\scaled', help="指定批次文件存储目录（不指定则使用临时目录）")
    parser.add_argument("--force_h5", action='store_true', help="强制使用HDF5格式输出，无论数据大小")
    parser.add_argument("--chunk_size", type=int, default=5000, help="PT合并时每次处理的最大样本数")
    args = parser.parse_args()

    main(args.input, args.output, args.max_seq_len, args.batch_size, 
         args.save_npz, args.max_workers, args.keep_temp, args.batch_dir,
         args.force_h5, args.chunk_size)   