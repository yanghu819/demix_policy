import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# 检查CUDA是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 生成双月形玩具数据
def generate_toy_data(n_samples=1000):
    np.random.seed(0)
    
    # 生成两个半圆形数据
    r = 10.0  # 半径
    theta = np.random.uniform(0, np.pi, n_samples // 2)
    
    # 第一个半圆
    x1 = r * np.cos(theta)
    y1 = r * np.sin(theta)
    
    # 第二个半圆（翻转并偏移）
    x2 = r * np.cos(theta + np.pi)
    y2 = r * np.sin(theta + np.pi) + 2  # 稍微偏移，使两个半圆不完全对称
    
    # 合并数据
    x = np.concatenate([x1, x2])
    y = np.concatenate([y1, y2])
    
    # 添加一些噪声
    x += np.random.normal(0, 0.5, n_samples)
    y += np.random.normal(0, 0.5, n_samples)
    
    # 转换为PyTorch张量并移至GPU
    data = torch.tensor(np.column_stack([x, y]), dtype=torch.float32).to(device)
    
    return data

# 时间嵌入
class TimeEmbedding(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.fc1 = nn.Linear(1, embed_dim)
        self.fc2 = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, t):
        # 确保t是二维的 [batch_size, 1]
        if t.dim() == 1:
            t = t.unsqueeze(1)
        
        h = F.silu(self.fc1(t))
        h = F.silu(self.fc2(h))
        return h

# 自适应层归一化
class AdaLN(nn.Module):
    def __init__(self, in_dim, embed_dim):
        super().__init__()
        self.in_dim = in_dim
        self.norm = nn.LayerNorm(in_dim)
        self.scale_shift = nn.Linear(embed_dim, in_dim * 2)
        
        # 初始化为零，以便开始时不进行缩放和偏移
        self.scale_shift.weight.data.zero_()
        self.scale_shift.bias.data.zero_()
    
    def forward(self, x, emb):
        # x: [batch_size, in_dim]
        # emb: [batch_size, embed_dim]
        
        x = self.norm(x)
        scale, shift = self.scale_shift(emb).chunk(2, dim=1)
        return x * (1 + scale) + shift

# 解耦流匹配模型
class DecoupledFlowMatching(nn.Module):
    def __init__(self, data_dim=2, hidden_dim=128, embed_dim=128, n_layers=8):
        super().__init__()
        self.data_dim = data_dim
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.n_layers = n_layers
        
        # 时间嵌入
        self.time_embed = TimeEmbedding(embed_dim)
        
        # 共享主干网络
        self.fc1 = nn.Linear(data_dim, hidden_dim)
        self.ada_ln1 = AdaLN(hidden_dim, embed_dim)
        
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ada_ln2 = AdaLN(hidden_dim, embed_dim)
        
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.ada_ln3 = AdaLN(hidden_dim, embed_dim)
        
        # 解耦的输出层
        self.gt_output = nn.Linear(hidden_dim, data_dim)
        self.noise_output = nn.Linear(hidden_dim, data_dim)
    
    def shared_forward(self, x, t):
        # 时间嵌入
        t_emb = self.time_embed(t)
        
        # 共享主干网络
        h = F.silu(self.ada_ln1(self.fc1(x), t_emb))
        h = F.silu(self.ada_ln2(self.fc2(h), t_emb))
        h = F.silu(self.ada_ln3(self.fc3(h), t_emb))
        
        return h
    
    def forward_layer(self, gt, noise, t):
        return self.forward(gt, noise, t)
    
    def forward(self, gt, noise, t):
        # 修复：确保t的形状正确
        t_expanded = t.view(-1, 1)  # 只在最后一个维度扩展
        
        # 混合输入 t*gt + (1-t)*noise
        mixed_input = t_expanded * gt + (1 - t_expanded) * noise
        
        # 通过共享主干网络
        h = self.shared_forward(mixed_input, t_expanded)
        
        # 通过解耦的输出层
        pred_gt = self.gt_output(h)
        pred_noise = self.noise_output(h)
        
        return pred_gt, pred_noise

# 训练函数
def train_model(model, data, n_epochs=200, batch_size=128):
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    n_samples = len(data)
    loss_history = []
    
    for epoch in range(n_epochs):
        total_loss = 0
        # 随机打乱数据
        indices = torch.randperm(n_samples, device=device)
        data_shuffled = data[indices]
        
        for i in range(0, n_samples, batch_size):
            optimizer.zero_grad()
            batch_data = data_shuffled[i:i+batch_size]
            batch_size_actual = len(batch_data)
            
            # 累积多层的损失
            layer_loss = 0
            layer_losses = []  # 记录每一层的损失
            
            # 初始GT和噪声
            gt = batch_data
            noise = torch.randn_like(gt, device=device)
            
            for layer in range(model.n_layers):
                # 为每一层采样时间参数t ~ U(0,1)
                t = torch.rand(batch_size_actual, device=device)
                
                # 前向传播
                pred_gt, pred_noise = model.forward_layer(gt, noise, t)
                
                # 计算损失（MSE）
                gt_loss = F.mse_loss(pred_gt, gt)
                noise_loss = F.mse_loss(pred_noise, noise)
                loss = gt_loss + noise_loss
                layer_loss += loss
                layer_losses.append(loss.item())  # 记录当前层的损失
                
                noise = torch.randn_like(gt, device=device)  # 新的随机噪声
            
            # 反向传播
            layer_loss.backward()
            optimizer.step()
            
            total_loss += layer_loss.item() * batch_size_actual
        
        scheduler.step()
        avg_loss = total_loss / n_samples
        loss_history.append(avg_loss)
        
        if epoch % 10 == 0 or epoch == n_epochs - 1:
            # 打印总损失和每一层的损失
            layer_avg_losses = [f"Layer {i+1}: {loss:.6f}" for i, loss in enumerate(layer_losses)]
            layer_loss_str = ", ".join(layer_avg_losses)
            print(f"Epoch {epoch+1}/{n_epochs}, Total Loss: {avg_loss:.6f}, LR: {scheduler.get_last_lr()[0]:.6f}")
            print(f"Layer Losses: {layer_loss_str}")
    
    return loss_history

# 推理函数
def inference(model, n_samples=100, data_dim=2):
    model.eval()
    with torch.no_grad():
        # 从纯噪声开始
        samples = torch.randn(n_samples, data_dim, device=device)
        
        # 存储每一层的样本
        all_samples = [samples.clone()]
        
        # 逐层推理，t值随层数单调增加
        for layer in range(model.n_layers):
            # t值从0.1开始，到0.9结束，确保单调增加
            t_val = 0.1 + 0.8 * (layer / (model.n_layers - 1))
            t = torch.ones(n_samples, device=device) * t_val
            
            # 生成新的噪声，噪声程度随层数减小
            noise_scale = 1.0 - t_val  # 噪声强度随t增加而减小
            noise = torch.randn_like(samples, device=device) * noise_scale
            
            # 前向传播，预测新的GT
            pred_gt, _ = model.forward_layer(samples, noise, t)
            
            # 更新样本
            samples = pred_gt
            all_samples.append(samples.clone())  # 存储当前层的样本
    
    # 将结果移回CPU用于可视化
    samples = samples.cpu()
    all_samples = [s.cpu() for s in all_samples]
    
    return samples, all_samples

# 可视化函数
def visualize_results(real_data, generated_samples, all_layer_samples=None, loss_history=None):
    # 确保数据在CPU上
    real_data = real_data.cpu()
    
    # 绘制真实数据和最终生成的样本
    plt.figure(figsize=(15, 5))
    
    # 绘制真实数据分布
    plt.subplot(131)
    plt.scatter(real_data[:, 0], real_data[:, 1], s=10, alpha=0.7)
    plt.title('真实数据分布')
    plt.grid(True)
    
    # 绘制生成的样本分布
    plt.subplot(132)
    plt.scatter(generated_samples[:, 0], generated_samples[:, 1], s=10, alpha=0.7, c='orange')
    plt.title('生成的样本')
    plt.grid(True)
    
    # 绘制损失曲线
    if loss_history is not None:
        plt.subplot(133)
        plt.plot(loss_history)
        plt.title('训练损失')
        plt.xlabel('轮次')
        plt.ylabel('损失')
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('./test_overview.png', dpi=300)
    plt.show()
    
    # 绘制每一层的生成结果
    if all_layer_samples is not None:
        n_layers = len(all_layer_samples)
        rows = (n_layers + 3) // 4  # 每行最多4个图
        plt.figure(figsize=(20, 5 * rows))
        
        for i, samples in enumerate(all_layer_samples):
            plt.subplot(rows, 4, i+1)
            plt.scatter(samples[:, 0], samples[:, 1], s=10, alpha=0.7, c='orange')
            if i == 0:
                plt.title('初始噪声')
            else:
                plt.title(f'第 {i} 层生成结果')
            plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('./test_layers.png', dpi=300)
        plt.show()

# 主函数
def main():
    # 生成toy数据
    data = generate_toy_data(n_samples=2000)
    
    # 创建模型并移至GPU
    model = DecoupledFlowMatching(data_dim=2, hidden_dim=128, embed_dim=128, n_layers=8).to(device)
    
    # 训练模型
    loss_history = train_model(model, data, n_epochs=200, batch_size=128)
    
    # 推理生成新样本，同时获取每一层的样本
    generated_samples, all_layer_samples = inference(model, n_samples=1000)
    
    # 可视化结果
    visualize_results(data, generated_samples, all_layer_samples, loss_history)

if __name__ == "__main__":
    main()
