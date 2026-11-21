import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18

# Paste the complete class definitions below in this order:
# 1. SiameseResNet18
class SiameseResNet18(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        backbone = resnet18(weights=None)
        self.encoder = nn.Sequential(*list(backbone.children())[:-2])
        self.head = nn.Sequential(
            nn.Conv2d(1024, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, num_classes, kernel_size=1),
        )

    def forward(self, image_t1, image_t2):
        features_t1 = self.encoder(image_t1)
        features_t2 = self.encoder(image_t2)
        difference = (features_t1 - features_t2).abs()
        return tuple(
            F.interpolate(
                self.head(torch.cat((features, difference), dim=1)),
                size=image_t1.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            for features in (features_t1, features_t2)
        )

# 2. SiameseTransformer
class SiameseTransformer(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, 192, kernel_size=16, stride=16) #  Patch Embedding
        self.position = nn.Parameter(torch.empty(1, 256, 192))
        nn.init.trunc_normal_(self.position, std=0.02)
        self.blocks = nn.Sequential(*[
            nn.TransformerEncoderLayer(
                d_model=192, nhead=6, dim_feedforward=768,
                batch_first=True, norm_first=True, activation="gelu",
            )
            for _ in range(4)
        ], nn.LayerNorm(192))
        self.head = nn.Sequential(
            nn.Conv2d(384, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, num_classes, 1),
        )
         # Transformer Encoder
    def encode(self, image):
        patches = self.patch_embed(image) # 
        tokens = patches.flatten(2).transpose(1, 2) # Flatten + Positional Embedding
        tokens = self.blocks(tokens + self.position)
        return tokens.transpose(1, 2).reshape_as(patches) # แปลงกลับเป็น Feature Map

    def forward(self, image1, image2):
        features1, features2 = self.encode(image1), self.encode(image2) #  Siamese Comparison เปรียบเทียบสองภาพ
        difference = (features1 - features2).abs()
        return tuple(   # Upsample กลับขนาดเดิม
            F.interpolate(
                self.head(torch.cat((features, difference), dim=1)),
                size=image1.shape[-2:], mode="bilinear", align_corners=False,
            )
            for features in (features1, features2)
        )


# 3. SiameseHybrid
class SiameseHybrid(SiameseTransformer):
    def __init__(self, num_classes):
        super().__init__(num_classes)

        cnn = resnet18(weights=None)
        self.patch_embed = nn.Sequential(
            cnn.conv1,
            cnn.bn1,
            cnn.relu,
            cnn.maxpool,
            cnn.layer1,
            cnn.layer2,
            cnn.layer3,
            nn.Conv2d(256, 192, kernel_size=1),
        )
