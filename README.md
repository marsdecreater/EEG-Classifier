

```markdown:d:\TSD\github\README.md
# EEG-Classifier

基于深度学习的脑电信号（EEG）手势分类模型。该模型使用改进的3D EEG分类器，结合时间卷积、空间注意力和GRU网络，实现对EEG信号的高效分类。

## 模型特点

- 使用时间卷积层捕获时间特征
- 采用空间注意力机制处理空间信息
- 使用双向GRU网络处理序列信息
- 实现自注意力机制增强特征表示
- 支持混合精度训练
- 实现早停机制防止过拟合
- 使用Weights & Biases进行实验追踪
- 支持数据增强（高斯噪声、时间偏移、振幅缩放）

## 环境要求

- Python 3.8+
- CUDA 支持（推荐）

详细依赖请查看 `requirements.txt`

## 数据集结构

数据集应按以下结构组织：

processed_dataset_v5/
    ├── gesture1/
    ├── gesture2/
    ├── gesture3/
    ...
    └── gesture10/


## 使用方法

1. 安装依赖：
```bash
pip install -r requirements.txt
```

2. 配置数据路径：
修改 `load_data` 函数中的数据路径：
```python
gesture_dir = f"你的数据集路径/{gesture}"
```

3. 运行训练：
```bash
python train.py
```

## 模型架构

- 输入层：原始EEG信号
- 特征提取：
  - 时间卷积层（Conv1d）
  - 空间注意力机制
  - 双向GRU网络
  - 自注意力层
- 输出层：全连接层

## 训练策略

- 使用 AdamW 优化器
- OneCycleLR 学习率调度
- 标签平滑
- 混合精度训练
- 早停机制（patience=10）

## 可视化

- 使用 Weights & Biases 追踪训练过程
- 生成混淆矩阵
- 输出详细的分类报告

## 许可证

[MIT License](LICENSE)




