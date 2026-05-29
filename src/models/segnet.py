"""
segnet.py — 自製缺陷分割網路(encoder-decoder + skip,多頭)

來源:scripts/train_stage4.py 的 DefectSegNetV4,架構**原樣保留**(這是自製模型)。
唯一改動:頭部不再寫死 2 / N_STATES / N_TYPES,改由 `src.schema` 推導 ——
新增/移除/改 head 只動 schema,模型自動跟上。

forward 回傳 `dict[head_key -> logits]`(取代原本寫死的三元組),讓 train/eval
可泛用地走 schema.HEADS,而不是硬綁 A/B/C 三個變數。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src import schema


class ConvBlock(nn.Module):
    """兩層 3x3 conv + BN + ReLU。"""

    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = F.relu(self.bn2(self.conv2(x)), inplace=True)
        return x


class DefectSegNet(nn.Module):
    """U-Net 式 encoder-decoder,共享 backbone + 每個 schema head 一個 1x1 conv。

    架構與 Stage 4 DefectSegNetV4 完全一致(4 層下採樣 + bottleneck + 4 層上採樣
    skip-concat),只把輸出頭改成 schema 驅動。
    """

    def __init__(self, base_c: int = 32, heads: tuple[schema.Head, ...] = schema.HEADS):
        super().__init__()
        c = base_c
        self.heads_spec = heads

        # encoder
        self.enc1 = ConvBlock(3, c)
        self.enc2 = ConvBlock(c, c * 2)
        self.enc3 = ConvBlock(c * 2, c * 4)
        self.enc4 = ConvBlock(c * 4, c * 8)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(c * 8, c * 8)

        # decoder
        self.up4 = nn.ConvTranspose2d(c * 8, c * 4, 2, stride=2)
        self.dec4 = ConvBlock(c * 4 + c * 8, c * 4)
        self.up3 = nn.ConvTranspose2d(c * 4, c * 2, 2, stride=2)
        self.dec3 = ConvBlock(c * 2 + c * 4, c * 2)
        self.up2 = nn.ConvTranspose2d(c * 2, c, 2, stride=2)
        self.dec2 = ConvBlock(c + c * 2, c)
        self.up1 = nn.ConvTranspose2d(c, c // 2, 2, stride=2)
        self.dec1 = ConvBlock(c // 2 + c, c // 2)

        # heads:從 schema 推導,key 與 schema 對齊
        self.head_convs = nn.ModuleDict(
            {h.key: nn.Conv2d(c // 2, h.num_classes, 1) for h in heads}
        )

    def forward(self, x) -> dict[str, torch.Tensor]:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return {key: conv(d1) for key, conv in self.head_convs.items()}


if __name__ == "__main__":
    # 形狀自檢:輸出每個 head 的通道數應等於 schema.output_channels()。
    net = DefectSegNet()
    dummy = torch.zeros(1, 3, 64, 64)
    out = net(dummy)
    print("output_channels(schema) =", schema.output_channels())
    for k, v in out.items():
        print(f"  head {k}: {tuple(v.shape)}")
    n_params = sum(p.numel() for p in net.parameters())
    print(f"params = {n_params:,}")
