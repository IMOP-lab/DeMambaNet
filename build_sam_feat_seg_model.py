import torch

from models.SamFeatSeg import SamFeatSeg, InternImage, MambaEncoder,SegDecoderCNN_add

def _build_feat_seg_model(num_classes=2):
    DrcM = SamFeatSeg(
       
        first=InternImage(channels=48),
        second=MambaEncoder(),
        seg_decoder_bl=SegDecoderCNN_add(),
        num_classes=num_classes,
    )
    return DrcM




model = _build_feat_seg_model( num_classes=2 )
model = model.cuda()
inputs = torch.randn(2, 3, 320, 320).cuda()
print(inputs.shape)
output = model(inputs)
print(output.shape)
