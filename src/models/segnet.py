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

    def __init__(self, base_c: int = 32, depth: int = 4,
                 heads: tuple[schema.Head, ...] = schema.HEADS):
        super().__init__()
        c = base_c
        self.heads_spec = heads
        self.depth = depth
        self.pool = nn.MaxPool2d(2)

        # encoder:第 i 層通道 c*2^i(i=0..depth-1),第 0 層吃 3 通道輸入。
        # depth=4 時等價於原寫死的 enc1..enc4(c, 2c, 4c, 8c)。
        ch = [c * (2 ** i) for i in range(depth)]
        self.encs = nn.ModuleList(
            [ConvBlock(3 if i == 0 else ch[i - 1], ch[i]) for i in range(depth)]
        )
        self.bottleneck = ConvBlock(ch[-1], ch[-1])

        # decoder:depth 個上採樣。第 t 步輸入通道 = ch[depth-1-t](t=0 來自 bottleneck),
        # concat 對應 enc skip(同通道);輸出 target:前 depth-1 步=ch[depth-2-t],最後一步=c//2。
        self.ups = nn.ModuleList()
        self.decs = nn.ModuleList()
        for t in range(depth):
            in_ch = ch[depth - 1 - t]
            skip_ch = ch[depth - 1 - t]
            out_ch = ch[depth - 2 - t] if t < depth - 1 else c // 2
            self.ups.append(nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2))
            self.decs.append(ConvBlock(out_ch + skip_ch, out_ch))

        # heads:從 schema 推導,key 與 schema 對齊
        self.head_convs = nn.ModuleDict(
            {h.key: nn.Conv2d(c // 2, h.num_classes, 1) for h in heads}
        )

    def forward(self, x) -> dict[str, torch.Tensor]:
        skips = []
        h = x
        for i, enc in enumerate(self.encs):
            h = enc(h if i == 0 else self.pool(h))
            skips.append(h)
        h = self.bottleneck(self.pool(h))
        for t, (up, dec) in enumerate(zip(self.ups, self.decs)):
            skip = skips[self.depth - 1 - t]
            h = dec(torch.cat([up(h), skip], dim=1))
        return {key: conv(h) for key, conv in self.head_convs.items()}


def load_state_dict_flexible(model: DefectSegNet, sd: dict) -> None:
    """載入 state_dict;若是舊版寫死命名(enc1/up4/dec4...)自動 remap 成 ModuleList 命名。

    舊→新(僅 depth=4 的舊 checkpoint):enc{1..4}→encs.{0..3}、up{4,3,2,1}→ups.{0..3}、
    dec{4,3,2,1}→decs.{0..3};bottleneck / head_convs 不變。
    """
    if any(k.startswith("encs.") for k in sd):
        model.load_state_dict(sd)
        return
    remap = {}
    enc_map = {"enc1": "encs.0", "enc2": "encs.1", "enc3": "encs.2", "enc4": "encs.3"}
    up_map = {"up4": "ups.0", "up3": "ups.1", "up2": "ups.2", "up1": "ups.3"}
    dec_map = {"dec4": "decs.0", "dec3": "decs.1", "dec2": "decs.2", "dec1": "decs.3"}
    pref = {**enc_map, **up_map, **dec_map}
    for k, v in sd.items():
        head = k.split(".", 1)[0]
        if head in pref:
            remap[k.replace(head, pref[head], 1)] = v
        else:
            remap[k] = v
    model.load_state_dict(remap)


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
