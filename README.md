# SSC Rolling 训练说明

本项目包含两个主要 rolling 训练脚本：

- `rolling_main.py`：普通 rolling 预测，默认一键跑 `enet` 和 `rf`
- `rolling_main_ablation.py`：RF 消融实验，支持 Linux + V100 上用 cuML GPU 随机森林

## 1. 准备数据

项目根目录至少需要：

```text
us_panel_cleaned.parquet
```

如果要跑 `rolling_main_ablation.py` 的 `v1/v2`，还需要：

```text
panel_winsorize_only.parquet
```

其中 `panel_winsorize_only.parquet` 是只做 winsorize、还没有 rank 的中间数据，不能从 `us_panel_cleaned.parquet` 反推回来。

## 2. Linux/V100 配环境

服务器 CUDA 版本是 12.2，推荐用 conda 创建 RAPIDS 环境：

```bash
conda create -n ssc-rapids -c rapidsai -c conda-forge -c nvidia \
    cuml cudf python=3.12 cuda-version=12.2

conda activate ssc-rapids
python -m pip install -r requirements.txt
```

检查 GPU/cuML：

```bash
nvidia-smi
python -c "from cuml.ensemble import RandomForestRegressor; print('cuML OK')"
```

## 3. 跑普通 rolling

默认跑 `enet` 和 CPU 版 `rf`：

```bash
python rolling_main.py
```

如果服务器上想跑 `enet` + GPU 版 RF：

```bash
SSC_MODELS=enet,rf_gpu python rolling_main.py
```

输出：

```text
preds/pred_enet_yearly.parquet
preds/pred_rf_yearly.parquet
preds/pred_rf_gpu_yearly.parquet
logs/training_log_enet_yearly.csv
logs/training_log_rf_yearly.csv
logs/training_log_rf_gpu_yearly.csv
```

实际会生成哪些文件取决于 `SSC_MODELS` 里选择了哪些模型。

## 4. 跑 RF 消融实验

`rolling_main_ablation.py` 只跑 RF，默认后端是 GPU：

```bash
SSC_RF_BACKEND=gpu python rolling_main_ablation.py
```

消融版本说明：

- `v0`：基准组，全部特征使用 rank 后数据，并自动选择 RF 超参数
- `v1`：把 `A_TOP10_FEATURES` 这 10 个特征替换为 winsorize 原始值，其余保持 rank，复用 `v0` 参数
- `v2`：把 `B_EXCLUSIVE_FEATURES` 替换为 winsorize 原始值，其余保持 rank，复用 `v0` 参数

推荐按顺序运行：

```bash
SSC_RF_BACKEND=gpu SSC_ABLATION_VERSION=0 python rolling_main_ablation.py
SSC_RF_BACKEND=gpu SSC_ABLATION_VERSION=1 python rolling_main_ablation.py
SSC_RF_BACKEND=gpu SSC_ABLATION_VERSION=2 python rolling_main_ablation.py
```

输出会放在单独目录：

```text
result_ablation_rf/preds/
result_ablation_rf/logs/
```

例如：

```text
result_ablation_rf/preds/pred_rf_yearly.parquet
result_ablation_rf/preds/pred_rf_yearly_v1.parquet
result_ablation_rf/preds/pred_rf_yearly_v2.parquet
result_ablation_rf/logs/training_log_rf_yearly.csv
result_ablation_rf/logs/training_log_rf_yearly_v1.csv
result_ablation_rf/logs/training_log_rf_yearly_v2.csv
```

如果 `panel_winsorize_only.parquet` 不在项目根目录，可以指定路径：

```bash
SSC_WINSORIZE_DATA_PATH=/path/to/panel_winsorize_only.parquet \
SSC_RF_BACKEND=gpu \
SSC_ABLATION_VERSION=1 \
python rolling_main_ablation.py
```

如果已有基准 RF 日志，也可以指定：

```bash
SSC_BASELINE_LOG_PATH=/path/to/training_log_rf_yearly.csv \
SSC_RF_BACKEND=gpu \
SSC_ABLATION_VERSION=1 \
python rolling_main_ablation.py
```

## 5. 注意事项

- `v1/v2` 必须先有 `v0` 的 `training_log_rf_yearly.csv`，因为它们要复用基准组每年的 RF 超参数。
- `panel_winsorize_only.parquet` 中原始特征缺失较多是正常现象，脚本会先按月中位数补缺，仍缺失的再填 `0.0`。
- 大数据和输出结果不要提交到 GitHub，`.gitignore` 已排除 `*.parquet`、`preds/`、`logs/`、`result_ablation_rf/` 等目录。
