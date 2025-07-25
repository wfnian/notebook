import math
import torch
from transformers import AutoModelForCausalLM
import matplotlib.pyplot as plt

# 加载模型
model = AutoModelForCausalLM.from_pretrained('thuml/sundial-base-128m', trust_remote_code=True)

# 构造输入序列
batch_size, lookback_length = 1, 800
t = torch.linspace(0, 8 * math.pi, steps=lookback_length)

# 多个频率和幅度的正余弦组合
seq = (
    0.6 * torch.sin(1.0 * t) +          # 基本波
    0.3 * torch.cos(2.5 * t) +          # 中频余弦波
    0.2 * torch.sin(4.0 * t + math.pi / 4) +  # 高频相位偏移正弦波
    0.1 * torch.cos(7.0 * t + math.pi / 3)    # 更高频的余弦波
)

# 扩展为 batch_size 的形状
seqs = seq.unsqueeze(0).repeat(batch_size, 1)  # 形状: (1, 800)
# 设置预测参数
forecast_length = 200
num_samples = 1

# 使用模型进行预测
output = model.generate(seqs, max_new_tokens=forecast_length, num_samples=num_samples)

# 检查输出维度
print(f"Output shape: {output.shape}")

# 假设输出形状是 (batch_size, forecast_length)
pred = output[0][0].detach().cpu().numpy()  # 取第一个 batch 的预测结果

# 构造扩展时间轴
t_future = torch.linspace(t[-1] + (t[1] - t[0]), t[-1] + forecast_length * (t[1] - t[0]), steps=forecast_length)

# 绘图
plt.figure(figsize=(12, 5))
plt.plot(t.numpy(), seqs[0].detach().cpu().numpy(), label='Input Sequence', color='blue')
plt.plot(t_future.numpy(), pred, label='Predicted Sequence', color='orange')
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Sundial Time Series Prediction')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
