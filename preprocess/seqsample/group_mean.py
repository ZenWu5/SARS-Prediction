import pandas as pd
from scipy import stats
import numpy as np
import os

import_dir = r"preprocess\mut2seq\output\merged_variant_output.csv"
output_dir = r"preprocess\seqsample\output\aim_seq_mean_log10Ka.csv"

# 检查输出目录是否存在，如果不存在则创建
if not os.path.exists(os.path.dirname(output_dir)):
    os.makedirs(os.path.dirname(output_dir))

df = pd.read_csv(import_dir)
df = df[df["log10Ka"].notna()]

def clip_and_mean(group):
    if len(group) < 3 or group["log10Ka"].std() == 0:
        # 数据太少或全相等，直接返回均值
        return pd.Series({
            "mean_log10Ka": group["log10Ka"].mean(),
            "count": len(group)
        })
    z = stats.zscore(group["log10Ka"])
    clipped = group[np.abs(z) < 2.5]
    return pd.Series({
        "mean_log10Ka": clipped["log10Ka"].mean(),
        "count": len(clipped)
    })

# 按 aim_seq 分组，clip 后求均值
result = df.groupby("aim_seq").apply(clip_and_mean).reset_index()

# 导出结果
result.to_csv(output_dir, index=False, encoding="utf-8-sig")
print(result.head())