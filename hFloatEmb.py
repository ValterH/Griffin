import argparse

import torch
import torch.nn as nn
from tqdm import tqdm

class SimpleRepeater(nn.Module):

    def __init__(self, dim=256) -> None:
        super().__init__()
        self.dim = dim
        self.register_buffer("scaler", torch.logspace(-2, 1, dim))
        self.register_buffer("floatscaler", torch.logspace(-2.5, 1, dim))
        self.floatenc = getfloatenc(dim)

    def forward(self, feat: torch.Tensor):
        assert feat.ndim == 1
        if torch.is_floating_point(feat): #, f"{feat.unique()} is categorical" # not true
            '''
            feat = feat.unsqueeze(1).to(torch.float) * self.floatscaler
            return torch.sin(feat)#torch.concat((x.unsqueeze(1), torch.sin(feat)), dim=-1)
            '''
            return self.floatenc(feat.unsqueeze(-1))#feat.unsqueeze(-1).expand(-1, self.dim)#
        else: # for categorical. Always positive
            feat = feat.unsqueeze(1).to(torch.float) * self.scaler
            return torch.cos(feat)

def getfloatenc(hiddim: int=256, train: bool=False):
    # 5000 1.2762317657470703 6.230667349882424e-05
    middim = int(hiddim**0.5+0.1)
    FloatEnc = nn.Sequential(nn.Linear(1, middim), nn.LayerNorm(middim, elementwise_affine=False), nn.SiLU(inplace=True), nn.Linear(middim, middim), nn.LayerNorm(middim, elementwise_affine=False), nn.SiLU(inplace=True), nn.Linear(middim, hiddim), nn.LayerNorm(hiddim, elementwise_affine=False), nn.SiLU(inplace=True),)
    if not train:
        FloatEnc.load_state_dict(torch.load(f"floatenc-{hiddim}.pt", map_location="cpu", weights_only=True))
        FloatEnc.eval()
        for p in FloatEnc.parameters():
            p.requires_grad_(False)
        FloatEnc = FloatEnc
    else:
        FloatEnc.train()
    return FloatEnc

def getfloatdec(hiddim: int=256, train: bool=False):
    # 5000 1.2762317657470703 6.230667349882424e-05
    FloatDec = nn.Sequential(nn.LayerNorm(hiddim, elementwise_affine=False), nn.Linear(hiddim, 1, bias=False))
    if not train:
        FloatDec.load_state_dict(torch.load(f"floatdec-{hiddim}.pt", map_location="cpu", weights_only=True))
        FloatDec.eval()
        for p in FloatDec.parameters():
            p.requires_grad_(False)
    else:
        FloatDec.train()
    return FloatDec

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--hiddim", type=int, default=512, help="hidden dimension for float encoder/decoder")
    args = argparser.parse_args()

    hiddim = args.hiddim
    floatdec = getfloatdec(hiddim, train=True)#nn.Sequential(nn.Linear(hiddim, hiddim, bias=False), nn.SiLU(inplace=True), nn.Linear(hiddim, 1, bias=False))
    floatenc = getfloatenc(hiddim, train=True)
    for p in floatdec.parameters():
        p.requires_grad_(True)
    for p in floatenc.parameters():
        p.requires_grad_(True)
    device = torch.device("cuda")
    floatdec, floatenc = floatdec.to(device), floatenc.to(device)
    optimizer = torch.optim.AdamW(list(floatdec.parameters())+list(floatenc.parameters()), lr=1e-3, weight_decay=1e-3)
    bestloss = 1000
    pbar = tqdm(total=100000)
    for i in range(100000):
        x = torch.randn((65536*16, 1), device=device)
        emb = floatenc(x)
        y = floatdec(emb) - x
        loss = y.square().mean() + 0.1 * emb.square().mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        pbar.set_description(f"loss: {loss.item():.4f}")
        pbar.update(1)
        if i%100==0:
            floatdec.eval()
            floatenc.eval()
            with torch.no_grad():
                x = torch.randn((65536*32, 1), device=device)
                y = floatdec(floatenc(x)) - x
                tloss = y.abs().mean().item() # y.abs().max().item()
            print(i, tloss, y.abs().max().item(), flush=True)
            if tloss < bestloss:
                print("save")
                torch.save(floatdec.state_dict(), f"floatdec-{hiddim}.pt")
                torch.save(floatenc.state_dict(), f"floatenc-{hiddim}.pt")
                bestloss = tloss
            floatdec.train()
            floatenc.train()
