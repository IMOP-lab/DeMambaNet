import torch

from .SamFeatSeg import SamFeatSeg, InternImage, MambaEncoder,SegDecoderCNN_add

def _build_feat_seg_model(

    checkpoint=1,
):
    DrcM = SamFeatSeg(
       
        first=InternImage(channels=48),
        second=MambaEncoder(),
        seg_decoder_bl=SegDecoderCNN_add(),
    )
    return DrcM




def build_DrcM(num_classes=14):
    # return _build_feat_seg_model(
    return _build_feat_seg_model()


sam_feat_seg_model_registry = build_DrcM
