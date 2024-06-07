import torch
import time
import timeit
from diffusers.models.attention import BasicTransformerBlock


torch.set_num_threads(2)
block = BasicTransformerBlock(
            320,
            8,
            40,
            dropout=0.0,
            cross_attention_dim=768,
            activation_fn="geglu",
            num_embeds_ada_norm=None,
            attention_bias=False,
            only_cross_attention=False,
            double_self_attention=False,
            upcast_attention=False,
            norm_type="layer_norm",
            norm_elementwise_affine=True,
            norm_eps=1e-05,
            attention_type="default",
        )

start = time.time()
block.forward(
        torch.randn((2, 4096, 320)),
        None,
        torch.randn((2, 77, 768)),
        None,
        None,
        None,
        None,
        None,
        )
print(f"{time.time() - start}")

#print(torch.__config__.parallel_info())
