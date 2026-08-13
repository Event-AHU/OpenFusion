# Ti / Te Profile Prediction

本项目使用宏观参数和二维光谱数据预测托卡马克离子温度（Ti）与电子温度（Te）剖面。程序入口为 `TiandTe.py`，模型采用 Transformer，并将 Ti、Te 剖面统一插值到 32 个径向位置。

## 环境依赖

建议使用 Python 3.6 及以上版本，并安装以下依赖：

```bash
pip install numpy pandas torch
```

## 数据准备

数据应为 CSV 文件，并放在同一个目录下。运行前请在 `TiandTe.py` 中修改数据路径：


CSV 中需要包含 Ti、Te 真实剖面与光谱数据，主要字段为：

- `Ti_out`：Ti 剖面
- `Te_out`：Te 剖面
- `Spec`：二维光谱数据
- 其余未被排除的列会作为宏观输入特征

## 运行方法

自动选择可用设备：

```bash
python 96.92.py
```

指定 GPU，例如使用第 0 张卡：

```bash
python TiandTe.py --gpu 0
```

程序将自动完成数据过滤与插值、训练集/验证集/测试集划分、模型训练、最优模型保存、测试指标计算及结果绘图。默认随机种子为 42，训练 400 个 epoch。

## 输出文件

运行结束后，主要生成：

- `best_model_attention.pth`：验证集上表现最好的模型权重
- `interpolation_raw_vs_smooth_attn.png`：原始剖面与插值结果
- `test_predictions_attn_Ti_soomth.png`：Ti 预测结果
- `test_predictions_attn_Te_soomth.png`：Te 预测结果
- `test_predictions_attn_RelError.png`：相对误差结果

终端中还会输出测试集的 MSE、RMSE、MAE 和 R2 指标。
