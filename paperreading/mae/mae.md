# Masked Autoencoders are Scalable Vision Learners

Tags: Computer Vision, Masked Image Modelling

Authors: Kaiming He et al.

Conferences/Journal: CVPR2022

Last edited time: September 4, 2023 8:59 PM

Masked Autoencoder (MAE) is an *asymmetric* autoencoder architecture wherein the encoder first operates on a subset of images (without mask tokens) and a lightweight decoder that reconstructs the original image from the latent representation and mask tokens. Concretely, the input image is masked with multiple patches at random, the decoder then reconstructs the missing patches (in the pixel space). MAE achieves *87.7%* accuracy when fine-tuned on ImageNet-1K using Vanilla VitHuge model.

![MAE architecture](Masked%20Autoencoders%20are%20Scalable%20Vision%20Learners%20ecac19f34caf47cebb9e1b43c54b9b76/Screen_Shot_2023-09-04_at_19.56.56.png)

**Masking:** The input image is first divided into multiple patches, as in Vision Transformer. Random patches is then removed. Random sampling with high masking ratio (\~75%) makes a task hard enough that it can not be solved with extrapolation alone.

``` python
def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore
```

**MAE encoder:** The encoder employs ViT but applied only on a small subset of patches from the input image. This allows training very large encoders.

``` python
def forward_encoder(self, x, mask_ratio):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore
```

**MAE decoder:** The input of the decoder are the set of encoded visible patches and mask tokens which indicate the missing part to be predicted. Positional embeddings are also aded to all tokens otherwise the mask tokens would have no location information in the image. The decoder reconstructs the input image by predicting the pixel values for each masked patch.

``` python
def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x
```

The encoder and decoder use different Transformer blocks.

**Results**

![](Masked%20Autoencoders%20are%20Scalable%20Vision%20Learners%20ecac19f34caf47cebb9e1b43c54b9b76/Screen_Shot_2023-09-04_at_20.40.49.png)

``` python
def mae_vit_large_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
```

![Comparisons with previous results on ImageNet1K](Masked%20Autoencoders%20are%20Scalable%20Vision%20Learners%20ecac19f34caf47cebb9e1b43c54b9b76/Screen_Shot_2023-09-04_at_20.41.17.png)

Comparisons with previous results on ImageNet1K

**Further Reading**

[Extracting and Composing Robust Features with Denoising Autoencoders](https://www.cs.toronto.edu/~larocheh/publications/icml-2008-denoising-autoencoders.pdf)

[Stacked Denoising Autoencoders: Learning Useful Representations in a Deep Network with a Local Denoising Criterion](https://www.jmlr.org/papers/volume11/vincent10a/vincent10a.pdf)

[An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/pdf/2010.11929.pdf)

[A Simple Framework for Contrastive Learning of Visual Representations](https://arxiv.org/pdf/2002.05709.pdf)

[Emerging Properties in Self-Supervised Vision Transformers](https://arxiv.org/pdf/2104.14294.pdf)

[R-MAE: Regions Meet Masked Autoencoders](https://arxiv.org/abs/2306.05411)
