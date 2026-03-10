import torch
import torch.nn as nn
import torch.nn.functional as F
from config import MODEL_PARAMS
import math

class FuzzyLayer(nn.Module):
    """
    模糊层：使用高斯隶属函数和模糊规则聚合来处理不精确的输入特征
    
    该层通过可学习的隶属函数参数（mean_value 和 sigma）来捕获输入空间中的
    模糊模式，特别适用于处理边界条件不精确的场景。
    
    Args:
        input_dim: 输入特征维度（对于 UniPINN，通常是 3，即 x, y, t）
        output_dim: 模糊特征输出维度（可学习的模糊规则数量）
    """
    def __init__(self, input_dim: int, output_dim: int):
        super(FuzzyLayer, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # 初始化模糊隶属函数的中心参数（mean_value）
        mean_value_weights = torch.Tensor(1, self.output_dim * self.input_dim)
        self.mean_value = nn.Parameter(mean_value_weights)
        
        # 初始化模糊隶属函数的宽度参数（sigma）
        sigma_weights = torch.Tensor(1, self.output_dim * self.input_dim)
        self.sigma = nn.Parameter(sigma_weights)
        
        # 初始化参数：mean_value 使用 Xavier 初始化，sigma 初始化为 1
        nn.init.xavier_uniform_(self.mean_value)
        nn.init.ones_(self.sigma)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        前向传播：计算模糊隶属度和模糊规则聚合
        
        Args:
            input: 输入张量，形状为 (batch_size, input_dim)
            
        Returns:
            模糊特征输出，形状为 (batch_size, output_dim)
        """
        # 模糊隶属度计算：对每个输入维度计算高斯隶属函数
        # input_expanded: (batch_size, output_dim * input_dim)
        input_expanded = input.repeat(1, self.output_dim)
        
        # 高斯隶属函数：exp(-(x - μ)² / σ²)
        fuzz_membership = torch.exp(
            -(input_expanded - self.mean_value).pow(2) / (self.sigma.pow(2) + 1e-8)
        )
        
        # 模糊规则聚合：使用乘积规则（product rule）聚合多维度隶属度
        # 将 fuzz_membership 重塑为 (batch_size, input_dim, output_dim)
        # 然后在 input_dim 维度上求乘积，得到 (batch_size, output_dim)
        fuzz_output = fuzz_membership.view(
            fuzz_membership.shape[0], self.input_dim, self.output_dim
        ).prod(dim=1)
        
        return fuzz_output

class SharedLayer(nn.Module):
    """
    共享层网络：提取多任务共享的特征表示
    
    支持可变输入维度，以兼容模糊层特征增强：
    - 不使用模糊层时：in_features = 3 (x, y, t)
    - 使用模糊层时：in_features = 3 + fuzzy_output_dim
    """
    def __init__(self, in_features: int = None):
        super().__init__()
        params = MODEL_PARAMS['shared_layer']
        
        # 如果未指定输入维度，从配置或默认值获取
        if in_features is None:
            # 检查是否使用模糊层
            try:
                from config import USE_FUZZY, FUZZY_OUTPUT_DIM
                if USE_FUZZY:
                    in_features = 3 + FUZZY_OUTPUT_DIM  # 原始输入 + 模糊特征
                else:
                    in_features = 3  # 仅原始输入 (x, y, t)
            except (ImportError, AttributeError):
                in_features = 3  # 默认值
        
        layers = []
        current_in_features = in_features
        
        # 减少层数，使用Tanh激活函数
        for _ in range(params['num_layers']):
            layers.extend([
                nn.Linear(current_in_features, params['num_neurons']),
                nn.Tanh()
            ])
            current_in_features = params['num_neurons']
        
        self.shared_layer = nn.Sequential(*layers)
    
    def forward(self, xyt):
        return self.shared_layer(xyt)

class DedicatedLayer(nn.Module):
    """专用层网络"""
    def __init__(self, task_type: str):
        super().__init__()
        params = MODEL_PARAMS['dedicated_layers'][task_type]
        
        layers = []
        in_features = MODEL_PARAMS['shared_layer']['num_neurons']
        
        # 减少层数，使用Tanh激活函数
        for _ in range(params['num_layers'] - 1):
            layers.extend([
                nn.Linear(in_features, params['num_neurons']),
                nn.Tanh()
            ])
            in_features = params['num_neurons']
        
        # 最后一层输出流函数和压力
        layers.append(nn.Linear(in_features, 2))
        
        self.dedicated_layer = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.dedicated_layer(x)

class SelfAttentionModule(nn.Module):
    """自注意力模块"""
    def __init__(self, attention_config):
        super().__init__()
        self.attention_dim = attention_config.get('attention_dim', 64)
        self.num_heads = attention_config.get('num_attention_heads', 8)
        self.dropout_rate = attention_config.get('attention_dropout', 0.1)
        
        # 多头注意力
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=self.attention_dim,
            num_heads=self.num_heads,
            dropout=self.dropout_rate,
            batch_first=True
        )
        
        # 层归一化
        self.layer_norm = nn.LayerNorm(self.attention_dim)
        
    def forward(self, x):
        """
        自注意力前向传播
        
        Args:
            x: 输入特征，形状为 (batch_size, feature_dim)
            
        Returns:
            增强后的特征，形状为 (batch_size, feature_dim)
        """
        # 保存残差连接
        residual = x
        
        # 调整维度以适配多头注意力 (batch_size, 1, feature_dim)
        x = x.unsqueeze(1)
        
        # 应用多头注意力
        attn_output, _ = self.multihead_attn(x, x, x)
        
        # 残差连接和层归一化
        output = self.layer_norm(attn_output.squeeze(1) + residual)
        
        return output

class Network(nn.Module):
    """
    统一多流型物理信息神经网络 (Unified Physics-Informed Neural Networks)
    
    该网络实现了跨流型注意力机制，能够同时学习多种Navier-Stokes流型
    支持消融实验，可以灵活控制注意力机制的各个组件
    
    Args:
        attention_config (dict, optional): 注意力机制配置字典
            - use_attention: 是否使用注意力机制
            - use_self_attention: 是否使用自注意力
            - use_cross_attention: 是否使用跨任务注意力
    """
    def __init__(self, attention_config=None):
        super().__init__()
        
        # 设置注意力配置，默认为完整配置
        if attention_config is None:
            from config import ATTENTION_CONFIG
            attention_config = ATTENTION_CONFIG
        
        self.use_attention = attention_config.get('use_attention', True)
        self.use_self_attention = attention_config.get('use_self_attention', True)
        self.use_cross_attention = attention_config.get('use_cross_attention', True)
        
        # 任务列表与任务嵌入（用于任务特定编码）
        self.task_list = ['lid_driven_cavity', 'pipe', 'pipe_flow', 'couette_flow']
        self.embed_dim = 8  # 任务嵌入维度，可按需调整
        self.task_embeddings = nn.ParameterDict({
            t: nn.Parameter(torch.randn(self.embed_dim)) for t in self.task_list
        })
        
        # 检查是否使用模糊层
        try:
            from config import USE_FUZZY, FUZZY_OUTPUT_DIM
            self.use_fuzzy = USE_FUZZY
            self.fuzzy_output_dim = FUZZY_OUTPUT_DIM if USE_FUZZY else 0
        except (ImportError, AttributeError):
            self.use_fuzzy = False
            self.fuzzy_output_dim = 0
        
        # 初始化模糊层（如果启用），并确定共享层输入维度
        base_dim = 3               # (x, y, t)
        periodic_dim = 4           # sin/cos x, sin/cos y
        z_dim = 3                  # 物理参数 [nu, rho, Re]
        embed_dim = self.embed_dim # 任务嵌入 e_i
        fuzzy_dim = self.fuzzy_output_dim if self.use_fuzzy else 0
        
        if self.use_fuzzy:
            self.fuzzy_layer = FuzzyLayer(input_dim=3, output_dim=self.fuzzy_output_dim)
            shared_in_features = base_dim + periodic_dim + z_dim + embed_dim + fuzzy_dim
        else:
            self.fuzzy_layer = None
            shared_in_features = base_dim + periodic_dim + z_dim + embed_dim
        
        # 初始化共享骨干网络（根据增强输入维度初始化）
        self.shared = SharedLayer(in_features=shared_in_features)
        
        # 初始化流型专用层
        self.lid_driven_cavity = DedicatedLayer('lid_driven_cavity')
        self.pipe = DedicatedLayer('pipe')
        self.couette_flow = DedicatedLayer('couette_flow')
        # self.shear_layer = DedicatedLayer('shear_layer')  # 已移除shear_layer
        
        # 任务列表（支持两种命名方式）
        self.task_names = ['lid_driven_cavity', 'pipe', 'pipe_flow', 'couette_flow']
        hidden_dim = MODEL_PARAMS['shared_layer']['num_neurons']
        
        # 根据配置初始化注意力组件
        if self.use_attention:
            if self.use_self_attention:
                self.self_attention = SelfAttentionModule(attention_config)
            if self.use_cross_attention:
                # 跨任务注意力的Q/K/V投影（每个任务一套投影）
                self.query_proj = nn.ModuleDict({t: nn.Linear(hidden_dim, hidden_dim) for t in self.task_names})
                self.key_proj = nn.ModuleDict({t: nn.Linear(hidden_dim, hidden_dim) for t in self.task_names})
                self.value_proj = nn.ModuleDict({t: nn.Linear(hidden_dim, hidden_dim) for t in self.task_names})
                
                # 缩放与融合系数
                self.attn_scale = math.sqrt(hidden_dim)
                self.attn_alpha = 0.5  # 可按需暴露到config
        
        # 初始化权重
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        """初始化网络权重"""
        if isinstance(m, nn.Linear):
            # 使用更简单的初始化方法
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    def apply_task_attention(self, shared_output: torch.Tensor, current_task: str) -> torch.Tensor:
        """跨任务注意力：用其他任务的K/V增强当前任务特征"""
        # 生成query（当前任务）
        query = self.query_proj[current_task](shared_output)  # [B, D]
        
        # 收集其他任务的K/V
        other_tasks = [t for t in self.task_names if t != current_task]
        if len(other_tasks) == 0:
            return shared_output
        keys = torch.stack([self.key_proj[t](shared_output) for t in other_tasks], dim=1)     # [B, T, D]
        values = torch.stack([self.value_proj[t](shared_output) for t in other_tasks], dim=1) # [B, T, D]
        
        # 注意力分数与权重
        attn_scores = torch.bmm(query.unsqueeze(1), keys.transpose(1, 2))  # [B, 1, T]
        attn_scores = attn_scores / self.attn_scale
        attn_weights = F.softmax(attn_scores, dim=-1)  # [B, 1, T]
        
        # 加权聚合其他任务的values
        weighted = torch.bmm(attn_weights, values).squeeze(1)  # [B, D]
        
        # 融合当前特征
        enhanced = shared_output + self.attn_alpha * weighted
        return enhanced
    
    def forward(self, xyt, flow_type: str):
        """
        前向传播
        
        Args:
            xyt: 输入张量，形状为(batch_size, 3)，对应归一化后的 (x', y', t')
            flow_type: 流动类型
            
        Returns:
            torch.Tensor: 输出张量，形状为(batch_size, 2)
        """
        # 拆分坐标
        x = xyt[:, 0:1]
        y = xyt[:, 1:2]
        t = xyt[:, 2:3]
        
        # 周期特征增强：sin/cos(2πx), sin/cos(2πy)
        sin_x = torch.sin(2 * math.pi * x)
        cos_x = torch.cos(2 * math.pi * x)
        sin_y = torch.sin(2 * math.pi * y)
        cos_y = torch.cos(2 * math.pi * y)
        periodic = torch.cat([sin_x, cos_x, sin_y, cos_y], dim=1)
        
        # 物理参数编码 z_i = [nu_i, rho_i, Re_i]
        from config import FLOW_PHYS_PARAMS
        base_name = 'pipe' if flow_type == 'pipe_flow' else flow_type
        phys = FLOW_PHYS_PARAMS.get(base_name, {})
        nu = float(phys.get('nu', 0.1))
        rho = float(phys.get('rho', 1.0))
        U_char = float(phys.get('U', 1.0))
        L_char = float(phys.get('L', 1.0))
        Re = U_char * L_char / (nu + 1e-8)
        
        z = xyt.new_tensor([nu, rho, Re]).unsqueeze(0).expand(xyt.size(0), -1)  # (B,3)
        
        # 任务嵌入 e_i
        embed_key = flow_type if flow_type in self.task_embeddings else base_name
        if embed_key not in self.task_embeddings:
            raise ValueError(f"不支持的流动类型用于任务嵌入: {flow_type}")
        e = self.task_embeddings[embed_key].unsqueeze(0).expand(xyt.size(0), -1)  # (B,embed_dim)
        
        # 模糊层特征增强（如果启用）
        if self.use_fuzzy and self.fuzzy_layer is not None:
            fuzzy_features = self.fuzzy_layer(xyt)  # (batch_size, fuzzy_output_dim)
            features = [x, y, t, periodic, z, e, fuzzy_features]
        else:
            features = [x, y, t, periodic, z, e]
        
        x_enhanced = torch.cat(features, dim=1)
        
        # 共享特征提取
        shared_output = self.shared(x_enhanced)
        
        # 注意力机制处理（如果启用）
        if self.use_attention:
            if self.use_self_attention:
                shared_output = self.self_attention(shared_output)
            if self.use_cross_attention:
                # 跨任务注意力增强
                if flow_type not in self.task_names:
                    raise ValueError(f"不支持的流动类型: {flow_type}")
                shared_output = self.apply_task_attention(shared_output, flow_type)
        
        # 流型特定预测
        if flow_type == 'lid_driven_cavity':
            return self.lid_driven_cavity(shared_output)
        elif flow_type == 'pipe' or flow_type == 'pipe_flow':
            return self.pipe(shared_output)
        elif flow_type == 'couette_flow':
            return self.couette_flow(shared_output)
        # elif flow_type == 'shear_layer':
        #     return self.shear_layer(shared_output)  # 已移除shear_layer
        else:
            raise ValueError(f"不支持的流动类型: {flow_type}")
    
    def to(self, device):
        """将模型移动到指定设备"""
        super().to(device)
        self.device = device
        return self 