import torch.nn as nn


class ECCVColorizer(nn.Module):
    """
    Zhang et al., "Colorful Image Colorization", ECCV 2016.

    Input : (B, 1, H, W)  — raw L channel in [0, 100]
    Output: (B, 2, H/4, W/4) — predicted a*b* in ~[-110, 110]

    Architecture mirrors the official richzhang/colorization repo so that
    state-dict keys are identical and pretrained weights load without remapping.
    """

    def __init__(self, norm_layer=nn.BatchNorm2d):
        super().__init__()

        self.model1 = nn.Sequential(
            nn.Conv2d(1,   64,  3, stride=1, padding=1, bias=True), nn.ReLU(True),
            nn.Conv2d(64,  64,  3, stride=2, padding=1, bias=True), nn.ReLU(True),
            norm_layer(64),
        )
        self.model2 = nn.Sequential(
            nn.Conv2d(64,  128, 3, stride=1, padding=1, bias=True), nn.ReLU(True),
            nn.Conv2d(128, 128, 3, stride=2, padding=1, bias=True), nn.ReLU(True),
            norm_layer(128),
        )
        self.model3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=1, padding=1, bias=True), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, stride=1, padding=1, bias=True), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, stride=2, padding=1, bias=True), nn.ReLU(True),
            norm_layer(256),
        )
        self.model4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, stride=1, padding=1, bias=True), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, stride=1, padding=1, bias=True), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, stride=1, padding=1, bias=True), nn.ReLU(True),
            norm_layer(512),
        )
        # Dilated convolutions keep spatial size while expanding receptive field
        self.model5 = nn.Sequential(
            nn.Conv2d(512, 512, 3, dilation=2, padding=2, bias=True), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, dilation=2, padding=2, bias=True), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, dilation=2, padding=2, bias=True), nn.ReLU(True),
            norm_layer(512),
        )
        self.model6 = nn.Sequential(
            nn.Conv2d(512, 512, 3, dilation=2, padding=2, bias=True), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, dilation=2, padding=2, bias=True), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, dilation=2, padding=2, bias=True), nn.ReLU(True),
            norm_layer(512),
        )
        self.model7 = nn.Sequential(
            nn.Conv2d(512, 512, 3, stride=1, padding=1, bias=True), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, stride=1, padding=1, bias=True), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, stride=1, padding=1, bias=True), nn.ReLU(True),
            norm_layer(512),
        )
        # Single ConvTranspose upsample → output is H/4 x W/4 relative to input
        self.model8 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1, bias=True), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, stride=1, padding=1, bias=True), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, stride=1, padding=1, bias=True), nn.ReLU(True),
            nn.Conv2d(256, 313, 1, bias=True),  # 313 quantised a*b* bins
        )

        self.softmax   = nn.Softmax(dim=1)
        self.model_out = nn.Conv2d(313, 2, 1, bias=False)  # soft-decode → 2-channel a*b*
        # Convenience upsample (not called in forward; use in postprocess or finetune head)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)

    # ------------------------------------------------------------------
    # Internal normalisation (mirrors the official implementation exactly)
    # ------------------------------------------------------------------
    def normalize_l(self, x):
        return (x - 50.0) / 100.0

    def unnormalize_ab(self, x):
        return x * 110.0

    def forward(self, x):
        x = self.normalize_l(x)
        x = self.model1(x)
        x = self.model2(x)
        x = self.model3(x)
        x = self.model4(x)
        x = self.model5(x)
        x = self.model6(x)
        x = self.model7(x)
        x = self.model8(x)                  # (B, 313, H/4, W/4)
        x = self.model_out(self.softmax(x)) # (B, 2,   H/4, W/4)
        return self.unnormalize_ab(x)
