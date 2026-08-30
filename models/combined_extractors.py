import torch, math
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

def _is_hwc(space):
    return len(space.shape)==3 and space.shape[2] in [1,3,4]

class ImpalaBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)
        self.res = nn.Conv2d(in_ch, out_ch, 1) if in_ch!=out_ch else nn.Identity()
    def forward(self, x):
        res = self.res(x)
        x = F.relu(self.conv1(x)); x = F.relu(self.conv2(x))
        # residual + pool
        x = x + F.interpolate(res, size=x.shape[2:], mode='nearest')
        return self.pool(x)

class ImpalaCNNExtractor(BaseFeaturesExtractor):
    """Impala-CNN 3 blocos (32,64,64) — mario-ds/src/impala_cnn.py"""
    def __init__(self, obs_space, features_dim=512):
        super().__init__(obs_space, features_dim)
        hwc=_is_hwc(obs_space); self.is_hwc=hwc
        n_in = int(obs_space.shape[2]) if hwc else int(obs_space.shape[0])
        self.block1 = ImpalaBlock(n_in, 32)
        self.block2 = ImpalaBlock(32, 64)
        self.block3 = ImpalaBlock(64, 64)
        with torch.no_grad():
            dummy = torch.zeros(1,64,64,n_in).permute(0,3,1,2) if hwc else torch.zeros(1,*obs_space.shape)
            x=self.block1(dummy); x=self.block2(x); x=self.block3(x)
            n_flat=x.view(1,-1).shape[1]
        self.fc = nn.Sequential(nn.Linear(n_flat, features_dim), nn.ReLU())
    def forward(self, o):
        if o.dtype==torch.uint8: o=o.float()/255.0
        elif o.max()>1.5: o=o/255.0
        if self.is_hwc and o.dim()==4 and o.shape[-1] in [1,3,4]: o=o.permute(0,3,1,2)
        x=self.block1(o); x=self.block2(x); x=self.block3(x)
        return self.fc(x.view(x.size(0),-1))

class ImpoolaCNNExtractor(BaseFeaturesExtractor):
    """Impoola-CNN GAP — Impala + Global Average Pooling (1.01M)"""
    def __init__(self, obs_space, features_dim=512):
        super().__init__(obs_space, features_dim)
        hwc=_is_hwc(obs_space); self.is_hwc=hwc
        n_in = int(obs_space.shape[2]) if hwc else int(obs_space.shape[0])
        self.block1 = ImpalaBlock(n_in, 32)
        self.block2 = ImpalaBlock(32, 64)
        self.block3 = ImpalaBlock(64, 64)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(64, features_dim), nn.ReLU())
    def forward(self, o):
        if o.dtype==torch.uint8: o=o.float()/255.0
        elif o.max()>1.5: o=o/255.0
        if self.is_hwc and o.dim()==4 and o.shape[-1] in [1,3,4]: o=o.permute(0,3,1,2)
        x=self.block1(o); x=self.block2(x); x=self.block3(x)
        x=self.gap(x).view(x.size(0),-1)
        return self.fc(x)

class LSTMAttentionExtractor(BaseFeaturesExtractor):
    """CNN 5 layers → BiLSTM 10 frames → Temporal Attention — Imitation-player:128 (simplificado para 1 frame com LSTM stateless)"""
    def __init__(self, obs_space, features_dim=512, hidden=256):
        super().__init__(obs_space, features_dim)
        hwc=_is_hwc(obs_space); self.is_hwc=hwc
        n_in = int(obs_space.shape[2]) if hwc else int(obs_space.shape[0])
        self.cnn = nn.Sequential(
            nn.Conv2d(n_in, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(), nn.AdaptiveAvgPool2d(4), nn.Flatten()
        )
        with torch.no_grad():
            dummy = torch.zeros(1,64,64,n_in).permute(0,3,1,2) if hwc else torch.zeros(1,*obs_space.shape)
            n_flat=self.cnn(dummy).shape[1]
        self.lstm = nn.LSTM(n_flat, hidden, batch_first=True, bidirectional=True)
        self.attn = nn.MultiheadAttention(hidden*2, num_heads=4, batch_first=True)
        self.fc = nn.Sequential(nn.Linear(hidden*2, features_dim), nn.ReLU())
    def forward(self, o):
        if o.dtype==torch.uint8: o=o.float()/255.0
        elif o.max()>1.5: o=o/255.0
        if self.is_hwc and o.dim()==4 and o.shape[-1] in [1,3,4]: o=o.permute(0,3,1,2)
        f=self.cnn(o)  # B x N
        # simular sequência de 4 com mesma feature (para manter LSTM)
        seq=f.unsqueeze(1).repeat(1,4,1)  # B x 4 x N
        lstm_out,_=self.lstm(seq)  # B x 4 x H*2
        attn_out,_=self.attn(lstm_out,lstm_out,lstm_out)
        pooled=attn_out.mean(dim=1)
        return self.fc(pooled)

class ViTExtractor(BaseFeaturesExtractor):
    """ViT 64 patches 16x16 → Transformer 4 layers — Imitation-player:136"""
    def __init__(self, obs_space, features_dim=512, patch=16, dim=128, depth=4, heads=4):
        super().__init__(obs_space, features_dim)
        hwc=_is_hwc(obs_space); self.is_hwc=hwc
        n_in = int(obs_space.shape[2]) if hwc else int(obs_space.shape[0])
        assert 64%patch==0
        self.patch=patch; self.dim=dim
        self.proj = nn.Conv2d(n_in, dim, kernel_size=patch, stride=patch)
        num_patches=(64//patch)**2
        self.pos_emb = nn.Parameter(torch.randn(1, num_patches, dim)*0.02)
        encoder_layer=nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=dim*4, batch_first=True)
        self.transformer=nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.fc=nn.Sequential(nn.Linear(dim, features_dim), nn.ReLU())
    def forward(self, o):
        if o.dtype==torch.uint8: o=o.float()/255.0
        elif o.max()>1.5: o=o/255.0
        if self.is_hwc and o.dim()==4 and o.shape[-1] in [1,3,4]: o=o.permute(0,3,1,2)
        x=self.proj(o)  # B x dim x H/p x W/p
        x=x.flatten(2).transpose(1,2)  # B x N x dim
        x=x+self.pos_emb
        x=self.transformer(x)
        x=x.mean(dim=1)
        return self.fc(x)

class ResNet18Extractor(BaseFeaturesExtractor):
    """ResNet-18 adaptado 3×64×64 — Imitation-player:155 (11.5M)"""
    def __init__(self, obs_space, features_dim=512):
        super().__init__(obs_space, features_dim)
        hwc=_is_hwc(obs_space); self.is_hwc=hwc
        n_in = int(obs_space.shape[2]) if hwc else int(obs_space.shape[0])
        # stem
        self.stem=nn.Sequential(nn.Conv2d(n_in,64,7,stride=2,padding=3), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(3,stride=2,padding=1))
        # 4 stages (2 blocks cada)
        def _block(in_c,out_c,stride):
            return nn.Sequential(
                nn.Conv2d(in_c,out_c,3,stride=stride,padding=1), nn.BatchNorm2d(out_c), nn.ReLU(),
                nn.Conv2d(out_c,out_c,3,padding=1), nn.BatchNorm2d(out_c)
            )
        self.layer1=nn.Sequential(_block(64,64,1), _block(64,64,1))
        self.layer2=nn.Sequential(_block(64,128,2), _block(128,128,1))
        self.layer3=nn.Sequential(_block(128,256,2), _block(256,256,1))
        self.layer4=nn.Sequential(_block(256,512,2), _block(512,512,1))
        self.gap=nn.AdaptiveAvgPool2d(1)
        self.fc=nn.Sequential(nn.Linear(512, features_dim), nn.ReLU())
        # skip projections
        self.skip2=nn.Sequential(nn.Conv2d(64,128,1,stride=2), nn.BatchNorm2d(128))
        self.skip3=nn.Sequential(nn.Conv2d(128,256,1,stride=2), nn.BatchNorm2d(256))
        self.skip4=nn.Sequential(nn.Conv2d(256,512,1,stride=2), nn.BatchNorm2d(512))
    def forward(self, o):
        if o.dtype==torch.uint8: o=o.float()/255.0
        elif o.max()>1.5: o=o/255.0
        if self.is_hwc and o.dim()==4 and o.shape[-1] in [1,3,4]: o=o.permute(0,3,1,2)
        x=self.stem(o)
        # layer1 (residual identity)
        for b in self.layer1:
            res=x; x=b(x); x=F.relu(x+res)
        # layer2
        res=self.skip2(x); x=self.layer2[0](x); x=self.layer2[1](x); x=F.relu(x+res)
        res=self.skip3(x); x=self.layer3[0](x); x=self.layer3[1](x); x=F.relu(x+res)
        res=self.skip4(x); x=self.layer4[0](x); x=self.layer4[1](x); x=F.relu(x+res)
        x=self.gap(x).view(x.size(0),-1)
        return self.fc(x)
