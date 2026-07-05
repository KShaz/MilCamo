import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class TransposedPatchEmbed(layers.Layer):
    """Transposed patch embedding for upsampling.

    Args:
        embed_dim: Output feature dimension
        patch_size: Patch size for transposed convolution
        stride: Stride for upsampling
    """

    def __init__(self, embed_dim, patch_size=2, stride=2, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.stride = stride

    def build(self, input_shape):
        self.proj = layers.Conv2DTranspose(
            self.embed_dim,
            kernel_size=self.patch_size,
            strides=self.stride,
            padding='same',
            use_bias=False,
            name="proj"
        )
        self.norm = layers.LayerNormalization(axis=-1, epsilon=1e-05, name="norm")
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        x = self.proj(inputs)
        x = self.norm(x)
        return x

class DecoderBlock(layers.Layer):
    """Vision Transformer decoder block with cross-attention and self-attention.

    Args:
        embed_dim: Feature dimension
        num_heads: Number of attention heads
        mlp_ratio: MLP expansion ratio
        dropout: Dropout rate
        attention_dropout: Attention dropout rate
        path_drop: Drop path rate
        layer_scale: Layer scaling coefficient
    """

    def __init__(
        self,
        embed_dim,
        num_heads=8,
        mlp_ratio=4.0,
        dropout=0.0,
        attention_dropout=0.0,
        path_drop=0.0,
        layer_scale=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.path_drop = path_drop
        self.layer_scale = layer_scale

    def build(self, input_shape):
        # Cross-attention (decoder features attend to encoder features)
        self.norm1 = layers.LayerNormalization(axis=-1, epsilon=1e-05, name="norm1")
        self.cross_attn = layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.embed_dim // self.num_heads,
            dropout=self.attention_dropout,
            name="cross_attn"
        )
        self.drop_path1 = DropPath(self.path_drop, name="drop_path1")

        # Self-attention
        self.norm2 = layers.LayerNormalization(axis=-1, epsilon=1e-05, name="norm2")
        self.self_attn = layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.embed_dim // self.num_heads,
            dropout=self.attention_dropout,
            name="self_attn"
        )
        self.drop_path2 = DropPath(self.path_drop, name="drop_path2")

        # MLP
        self.norm3 = layers.LayerNormalization(axis=-1, epsilon=1e-05, name="norm3")
        self.mlp = MLP(
            hidden_features=int(self.embed_dim * self.mlp_ratio),
            dropout=self.dropout,
            name="mlp"
        )
        self.drop_path3 = DropPath(self.path_drop, name="drop_path3")

        # Layer scale
        if self.layer_scale is not None:
            self.gamma1 = self.add_weight(
                name="gamma1",
                shape=[self.embed_dim],
                initializer=keras.initializers.Constant(self.layer_scale),
                trainable=True,
                dtype=self.dtype,
            )
            self.gamma2 = self.add_weight(
                name="gamma2",
                shape=[self.embed_dim],
                initializer=keras.initializers.Constant(self.layer_scale),
                trainable=True,
                dtype=self.dtype,
            )
            self.gamma3 = self.add_weight(
                name="gamma3",
                shape=[self.embed_dim],
                initializer=keras.initializers.Constant(self.layer_scale),
                trainable=True,
                dtype=self.dtype,
            )
        else:
            self.gamma1 = 1.0
            self.gamma2 = 1.0
            self.gamma3 = 1.0

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        """
        Args:
            inputs: [decoder_features, encoder_features]
            decoder_features: Current decoder features (B, H, W, C)
            encoder_features: Skip connection features from encoder (B, H, W, C)
        """
        decoder_features, encoder_features = inputs

        # Reshape for attention (B, H, W, C) -> (B, H*W, C)
        B, H, W, C = tf.shape(decoder_features)[0], tf.shape(decoder_features)[1], tf.shape(decoder_features)[2], tf.shape(decoder_features)[3]
        decoder_flat = tf.reshape(decoder_features, [B, H * W, C])

        B_enc, H_enc, W_enc, C_enc = tf.shape(encoder_features)[0], tf.shape(encoder_features)[1], tf.shape(encoder_features)[2], tf.shape(encoder_features)[3]
        encoder_flat = tf.reshape(encoder_features, [B_enc, H_enc * W_enc, C_enc])

        # Cross-attention: decoder queries attend to encoder keys/values
        decoder_norm1 = self.norm1(decoder_flat)
        cross_attn_out = self.cross_attn(
            query=decoder_norm1,
            key=encoder_flat,
            value=encoder_flat
        )
        decoder_flat = decoder_flat + self.drop_path1(cross_attn_out * self.gamma1)

        # Self-attention
        decoder_norm2 = self.norm2(decoder_flat)
        self_attn_out = self.self_attn(
            query=decoder_norm2,
            key=decoder_norm2,
            value=decoder_norm2
        )
        decoder_flat = decoder_flat + self.drop_path2(self_attn_out * self.gamma2)

        # MLP
        decoder_norm3 = self.norm3(decoder_flat)
        mlp_out = self.mlp(decoder_norm3)
        decoder_flat = decoder_flat + self.drop_path3(mlp_out * self.gamma3)

        # Reshape back to spatial format
        output = tf.reshape(decoder_flat, [B, H, W, C])

        return output
