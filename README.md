# Encoder + TabM 光伏超短期预测

`encoder_tabm4pv.py` 采用和旧版 `tabm4pv.py` 相同的固定配置方式，不使用
命令行参数。运行前直接修改脚本顶部的 `Experiment configuration` 常量。

模型流程：

```text
最近 96 点标幺功率
  -> 1D CNN ShapeEncoder
  -> 32 维形态特征
  -> 拼接第 16 点未来 NWP 和周期时间特征
  -> rtdl_num_embeddings.LinearReLUEmbeddings
  -> TabM
  -> 第 16 点标幺功率
  -> 乘场站容量恢复实际功率
```

Encoder 和 TabM 使用同一个预测损失联合训练。训练时每个 TabM ensemble
member 分别拟合标签，推理时才取成员均值。

## 主要配置

先在 `encoder_tabm4pv.py` 顶部修改：

```python
DATA_ROOT_PATH = "/data/hjs/1219_report_onv8/"
DATA_FILE_SUFFIX = "_v1.parquet"
DATA_FILE_PREFIX = ""

CAPACITY_COL = None
DEFAULT_CAPACITY = 500.0
```

多场站 OOD 训练应将 `CAPACITY_COL` 改成 parquet 中真实的装机容量列。

默认采用旧版时间切分：

```python
SPLIT_MODE = "time"
TRAIN_YEAR = 2024
TRAIN_START_MONTH = 9
TEST_YEAR = 2025
```

完整留出场站时改成：

```python
SPLIT_MODE = "station"
OOD_VALIDATION_FILE_PREFIXES = ("mkv81",)
OOD_TEST_FILE_PREFIXES = ("mkv82",)
```

其余不匹配以上前缀的场站文件将作为训练集。

## 运行

```bash
pip install -r requirements.txt
python encoder_tabm4pv.py
```

输出保存在 `outputs/encoder_tabm/`：

- `best_model.pt`
- `future_covariate_transformer.joblib`
- `test_predictions.parquet`
- `test_metrics_by_station.csv`
- `metrics.json`

第一轮建议保持：

```python
INVARIANCE_LOSS_WEIGHT = 0.0
```

先单独验证 Encoder 是否改善 OOD，再尝试 `0.05` 左右的幅值一致性约束。
