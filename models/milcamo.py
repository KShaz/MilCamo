import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras import ops
import numpy as np

from .encoder import GCViT
from .decoder import DecoderBlock, TransposedPatchEmbed


def segmentationModelFunction():
    config = {
        "window_size": (7,7,14,7),
        "embed_dim":64,
        "depths":(2,2,6,2),
        "num_heads":(2,4,8,16),
        "mlp_ratio":3.0,
        "path_drop":0.2,
    }
    ckpt_link='https://github.com/awsaf49/gcvit-tf/releases/download/v1.1.6/gcvitxxtiny.keras'
    backbone=GCViT(**config)
    backbone(ops.array(np.random.uniform(size=(1,224,224,3))))
    ckpt_path=keras.utils.get_file(ckpt_link.split('/')[-1], ckpt_link)
    backbone.load_weights(ckpt_path)

    inputs=layers.Input(shape=(224,224,3))
    output, skips = backbone(inputs)

    x=DecoderBlock(embed_dim=512,num_heads=8)([output,skips['3']])
    x=TransposedPatchEmbed(embed_dim=512,patch_size=2,stride=2)(x)
    x=DecoderBlock(embed_dim=256,num_heads=8)([x,skips['2']])
    x=TransposedPatchEmbed(embed_dim=256,patch_size=2,stride=2)(x)
    x=DecoderBlock(embed_dim=128,num_heads=8)([x,skips['1']])
    x=TransposedPatchEmbed(embed_dim=128,patch_size=2,stride=2)(x)
    x=DecoderBlock(embed_dim=64,num_heads=8)([x,skips['0']])
    x=TransposedPatchEmbed(embed_dim=64,patch_size=2,stride=2)(x)
    x=TransposedPatchEmbed(embed_dim=32,patch_size=2,stride=2)(x)
    x=TransposedPatchEmbed(embed_dim=16,patch_size=2,stride=2)(x)
    outputs=layers.Conv2D(1,1,padding='same',activation='sigmoid')(x)
    return keras.Model(inputs,outputs,name='MilCamo')
