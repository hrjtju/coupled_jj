"""
Consists of FeatureExtractor and ParamPredictor Network.

"""
import torch
from torch import Tensor, nn
import torch.nn.init as init


class PositionalEncoding(nn.Module):
    """
    Sinusoidal Position Embedding for Transformer Models
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 创建位置编码矩阵 [max_len, d_model]
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        
        # 计算 div_term: 10000^(2i/d_model)
        div_term = torch.exp(torch.arange(0, d_model, 2) * 
                            (-torch.log(torch.tensor(10000.0)) / d_model))
        
        # 偶数维度使用 sin，奇数维度使用 cos
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            # d_model 为奇数时，处理最后一个奇数维度
            pe[:, 1::2] = torch.cos(position * div_term[:pe[:, 1::2].shape[1]])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        
        # 注册为 buffer（不作为可学习参数，但会保存到 state_dict）
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, d_model]
        Returns:
            x: [batch, seq_len, d_model] (添加了位置编码)
        """
        seq_len = x.size(1)
        x = x + self.pe[:seq_len, :]
        return self.dropout(x)


class FourierFeatureEncoder(nn.Module):
    def __init__(self, n_frequencies=32):
        super().__init__()
        
        # 对相位变量使用随机傅里叶特征
        self.B = (torch.randn(2, n_frequencies) * 2 * torch.pi).cuda()  # 2个相位
        
    def forward(self, traj):
        phi = traj[..., [0, 2]].cuda()  # phi1, phi2
        v = traj[..., [1, 3]].cuda()    # v1, v2
        
        # print(traj.shape)
        # print(phi.shape)
        
        # Fourier嵌入: [sin(2π B phi), cos(2π B phi)]
        mul = phi @ self.B
        phi_proj = torch.cat([torch.sin(mul), 
                              torch.cos(mul)], dim=-1)
        
        # print(phi_proj.shape)
        # print(torch.cat([phi_proj, v], dim=-1).shape)
        
        # 与电压拼接后送入Transformer
        return torch.cat([phi_proj, v], dim=-1)

class FeatureExtractor(nn.Module):
    """
    FeatureExtractor
    """

    def __init__(self, in_dim: int, hidden_dim: int, n_layer: int,
                 out_dim: int = 256, activation: str = 'elu', 
                 nhead: int = 8, dropout: float = 0.1):
        
        super().__init__()

        self.type = 'Transformer'
        self.activation = activation

        self.init_param = [in_dim, hidden_dim, n_layer, out_dim]
        print('init_param', self.init_param)

        self.n_layer = n_layer
        self.hidden_dim = hidden_dim
        self.nhead = nhead

        # Part 1: Transformer Encoder with Positional Encoding
        # 输入投影层：将 in_dim 映射到 hidden_dim
        self.fourier_feature = FourierFeatureEncoder().cuda()
        self.conv1 = nn.Conv1d(66, in_dim, kernel_size=3, stride=2, padding=0).cuda()
        self.bn = nn.BatchNorm1d(in_dim).cuda()
        self.input_projection = nn.Linear(in_dim, hidden_dim).cuda()
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(hidden_dim, max_len=5000, dropout=dropout)
        
        # Transformer Encoder Layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,  # 通常设置为 hidden_dim * 4
            dropout=dropout,
            batch_first=True,
        )
        
        # Transformer Encoder (多层堆叠)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layer
        )
        
        # Layer Normalization (Transformer 后的归一化)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        input_t = torch.randn((1, 2000, 4)).cuda()
        x_conv = self.input_projection(self.conv1(self.fourier_feature(input_t).transpose(1, 2).float()).transpose(1, 2))
        t_dim = x_conv.shape[1] * x_conv.shape[2]
        self.down_proj = nn.Linear(t_dim, out_dim)
        
        # 初始化权重
        self.init_weights()

    def init_weights(self):
        """初始化网络权重"""
        # 输入投影层初始化
        init.xavier_uniform_(self.input_projection.weight)
        if self.input_projection.bias is not None:
            init.zeros_(self.input_projection.bias)
        
        # 线性层使用 Xavier 初始化
        for m in self.modules():
            if isinstance(m, nn.Linear) and m != self.input_projection:
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.zeros_(m.bias)
    def forward(self, x: torch.Tensor):
        """
        前向传播。

        Args:
            x: 轨迹数据 [batch, max_N, state_dim]
            l: 真实长度 [batch, 1]
            h: 时间跨度 [batch, 1]
            z: mask [batch, max_N, 1]

        Returns:
            pred_params: [batch, num_params]
        """
        # Part 1: Transformer Encoder with Positional Encoding
        # 1. 输入投影: [batch, seq_len, in_dim] -> [batch, seq_len, hidden_dim]
        
        b, l, c = x.shape
        x_conv = self.bn(self.conv1(self.fourier_feature(x).transpose(1, 2).float()))
        x_input = self.input_projection(x_conv.transpose(1, 2))
        
        # 2. 添加位置编码
        x_encoded = self.pos_encoder(x_input)
        
        # 4. Transformer Encoder
        # src_key_padding_mask: [batch, seq_len]，True 表示 mask 掉
        _out = self.transformer_encoder(x_encoded)
        
        # is this place appropriate to add a LayerNorm ?
        _out: Tensor = self.layer_norm(_out)
        
        return self.down_proj(_out.flatten(1))


class ParamPredictor(nn.Module):
    """
    ParamPredictor
    """
    def __init__(self, in_dim: int, hid_dim: int, n_params: int):
        super().__init__()
        
        self.linears = nn.ModuleList()
        
        for k in range(3):
            if k == 0:
                self.linears.append(nn.Linear(in_dim, hid_dim))
            else:
                self.linears.append(nn.Linear(hid_dim, hid_dim))
            self.linears.append(nn.ELU(inplace=True))
        
        # 输出层
        self.linears.append(nn.Linear(hid_dim, n_params))
    
    def forward(self, x: Tensor):
        for layer in self.linears:
            x = layer(x)
        return x

if __name__ == "__main__":
    # test the classes and functions.
    
    input_t = torch.randn((1, 1000, 4))
    feature_ex = FeatureExtractor(in_dim=4, hidden_dim=20, n_layer=2)
    predictor = ParamPredictor(in_dim=256, hid_dim=128, n_params=8)
    feature = feature_ex(input_t)
    print(f"feature shape: {feature.shape=}")
    pred_params = predictor(feature)
    print(f"pred_params shape: {pred_params.shape=}\n{pred_params=}")
