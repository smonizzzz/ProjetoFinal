import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class ConvBnRelu(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, stride=1, padding=1, bias=False):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, stride, padding, bias=bias),
            nn.BatchNorm2d(out_ch, momentum=0.1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch, momentum=0.1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch, momentum=0.1)
        self.relu  = nn.ReLU(inplace=True)
        self.downsample = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch, momentum=0.1),
            )

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample:
            residual = self.downsample(x)
        return self.relu(out + residual)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_ch, mid_ch, stride=1):
        super().__init__()
        out_ch     = mid_ch * self.expansion
        self.conv1 = nn.Conv2d(in_ch, mid_ch, 1, bias=False)
        self.bn1   = nn.BatchNorm2d(mid_ch, momentum=0.1)
        self.conv2 = nn.Conv2d(mid_ch, mid_ch, 3, stride, 1, bias=False)
        self.bn2   = nn.BatchNorm2d(mid_ch, momentum=0.1)
        self.conv3 = nn.Conv2d(mid_ch, out_ch, 1, bias=False)
        self.bn3   = nn.BatchNorm2d(out_ch, momentum=0.1)
        self.relu  = nn.ReLU(inplace=True)
        self.downsample = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
            nn.BatchNorm2d(out_ch, momentum=0.1),
        )

    def forward(self, x):
        residual = self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return self.relu(out + residual)


class FusionLayer(nn.Module):
    def __init__(self, num_branches, channels):
        super().__init__()
        self.num_branches = num_branches
        self.fuse_layers  = nn.ModuleList()
        for i in range(num_branches):
            row = nn.ModuleList()
            for j in range(num_branches):
                if j > i:
                    row.append(nn.Sequential(
                        nn.Conv2d(channels[j], channels[i], 1, bias=False),
                        nn.BatchNorm2d(channels[i], momentum=0.1),
                    ))
                elif j < i:
                    convs = []
                    for k in range(i - j - 1):
                        convs += [
                            nn.Conv2d(channels[j], channels[j], 3, 2, 1, bias=False),
                            nn.BatchNorm2d(channels[j], momentum=0.1),
                            nn.ReLU(inplace=True),
                        ]
                    convs += [
                        nn.Conv2d(channels[j], channels[i], 3, 2, 1, bias=False),
                        nn.BatchNorm2d(channels[i], momentum=0.1),
                    ]
                    row.append(nn.Sequential(*convs))
                else:
                    row.append(nn.Identity())
            self.fuse_layers.append(row)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, branches):
        fused = []
        for i in range(self.num_branches):
            x = self.fuse_layers[i][0](branches[0])
            for j in range(1, self.num_branches):
                if j > i:
                    y = self.fuse_layers[i][j](branches[j])
                    y = F.interpolate(y, size=branches[i].shape[2:], mode="bilinear", align_corners=False)
                    x = x + y
                else:
                    x = x + self.fuse_layers[i][j](branches[j])
            fused.append(self.relu(x))
        return fused


class HRStage(nn.Module):
    def __init__(self, num_branches, channels, num_blocks=4, num_modules=1):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(*[BasicBlock(channels[i], channels[i]) for _ in range(num_blocks)])
            for i in range(num_branches)
        ])
        self.fusions = nn.ModuleList([FusionLayer(num_branches, channels) for _ in range(num_modules)])

    def forward(self, x):
        for i, branch in enumerate(self.branches):
            x[i] = branch(x[i])
        for fusion in self.fusions:
            x = fusion(x)
        return x


class HRNetW32(nn.Module):
    def __init__(self, num_outputs=3):
        super().__init__()
        self.stem = nn.Sequential(
            ConvBnRelu(3, 64, kernel=3, stride=2, padding=1),
            ConvBnRelu(64, 64, kernel=3, stride=2, padding=1),
            Bottleneck(64, 64),
            Bottleneck(256, 64),
            Bottleneck(256, 64),
            Bottleneck(256, 64),
        )
        self.trans1 = nn.ModuleList([
            nn.Sequential(nn.Conv2d(256, 32, 3, 1, 1, bias=False), nn.BatchNorm2d(32, momentum=0.1), nn.ReLU(inplace=True)),
            nn.Sequential(nn.Conv2d(256, 64, 3, 2, 1, bias=False), nn.BatchNorm2d(64, momentum=0.1), nn.ReLU(inplace=True)),
        ])
        self.stage2 = HRStage(2, [32, 64], num_blocks=4, num_modules=1)
        self.trans2 = nn.ModuleList([
            nn.Identity(), nn.Identity(),
            nn.Sequential(nn.Conv2d(64, 128, 3, 2, 1, bias=False), nn.BatchNorm2d(128, momentum=0.1), nn.ReLU(inplace=True)),
        ])
        self.stage3 = HRStage(3, [32, 64, 128], num_blocks=4, num_modules=4)
        self.trans3 = nn.ModuleList([
            nn.Identity(), nn.Identity(), nn.Identity(),
            nn.Sequential(nn.Conv2d(128, 256, 3, 2, 1, bias=False), nn.BatchNorm2d(256, momentum=0.1), nn.ReLU(inplace=True)),
        ])
        self.stage4 = HRStage(4, [32, 64, 128, 256], num_blocks=4, num_modules=3)

        # Cabeça de regressão para prever ângulos de Cobb
        self.regression_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32 + 64 + 128 + 256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_outputs),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x)
        branches = [t(x) for t in self.trans1]
        branches = self.stage2(branches)
        branches = [self.trans2[0](branches[0]), self.trans2[1](branches[1]), self.trans2[2](branches[1])]
        branches = self.stage3(branches)
        branches = [self.trans3[0](branches[0]), self.trans3[1](branches[1]), self.trans3[2](branches[2]), self.trans3[3](branches[2])]
        branches = self.stage4(branches)

        # Fundir todos os branches para regressão
        target_size = branches[0].shape[2:]
        fused = torch.cat([
            branches[0],
            F.interpolate(branches[1], size=target_size, mode="bilinear", align_corners=False),
            F.interpolate(branches[2], size=target_size, mode="bilinear", align_corners=False),
            F.interpolate(branches[3], size=target_size, mode="bilinear", align_corners=False),
        ], dim=1)

        return self.regression_head(fused)  # (B, 3)


def build_model(arch="hrnet", num_outputs=3, pretrained=False, pretrained_path=None):
    if arch == "hrnet":
        model = HRNetW32(num_outputs=num_outputs)
    else:
        raise ValueError(f"Arquitectura desconhecida: {arch!r}. Use 'hrnet'.")

    if pretrained and pretrained_path:
        state      = torch.load(pretrained_path, map_location="cpu")
        model_state = model.state_dict()
        filtered   = {k: v for k, v in state.items() if k in model_state and v.shape == model_state[k].shape}
        model.load_state_dict(filtered, strict=False)
        print(f"[build_model] Carregados {len(filtered)} pesos pré-treinados")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[build_model] arch={arch} | parâmetros: {n_params:,}")
    return model
