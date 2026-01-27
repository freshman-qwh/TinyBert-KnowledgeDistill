import pandas as pd
import os


def convert_parquet_to_tsv(input_file, output_file):
    # 1. 读取 Parquet 文件
    # SST-2 常见的列名是 'sentence' 和 'label'
    df = pd.read_parquet(input_file)

    # 2. 如果数据集中有 'idx' 列，通常在训练时不需要，可以保留也可以删除
    # TinyBERT 的增强脚本通常只需要 sentence 和 label
    if 'idx' in df.columns:
        df = df.drop(columns=['idx'])

    # 3. 确保数据保存为 TSV 格式
    # index=False 不保存行索引
    # sep='\t' 指定制表符分隔
    # quoting=3 表示不使用引号包裹，这是 BERT 原始数据的标准格式
    df.to_csv(output_file, sep='\t', index=False, encoding='utf-8')

    print(f"转换完成: {input_file} -> {output_file}")
    print(f"样本数量: {len(df)}")
    print(f"数据预览:\n{df.head(2)}")


# 使用示例
# 假设你的文件名为 train-00000-of-00001.parquet
convert_parquet_to_tsv('TinyBERT/SST-2/parquet_format/train-00000-of-00001.parquet', 'SST-2/train.tsv')
convert_parquet_to_tsv('TinyBERT/SST-2/parquet_format/validation-00000-of-00001.parquet', 'SST-2/dev.tsv')
convert_parquet_to_tsv('TinyBERT/SST-2/parquet_format/test-00000-of-00001.parquet', 'SST-2/test.tsv')