from __future__ import annotations

from abc import ABC, abstractmethod
from tkinter import Image
from tkinter import Image
from typing import Any, NamedTuple

import torchvision
from transformers import AutoModel, AutoProcessor

import torch

class EncodeDenseOutput(NamedTuple):
    """Output from encode_dense - all tensors needed for Grad-ECLIP."""
    embeddings: torch.Tensor       # (batch, seq_len, hidden) - final with projection
    q_out: torch.Tensor            # (seq_len, batch, hidden) - Q from last attention
    k_out: torch.Tensor            # (seq_len, batch, hidden) - K from last attention
    v: torch.Tensor                # (seq_len, batch, hidden) - V from last attention
    att_output: torch.Tensor       # (seq_len, batch, hidden) - attention before residual
    patch_map_size: tuple          # (H, W) - spatial grid of patches
    classic_output: torch.Tensor   # (batch, seq_len, hidden) - final output without projection (for sanity checks)


class ClipModel(ABC):
    """Common API for CLIP-like models used in experiments."""

    def __init__(
        self,
        model_id: str,
        device: str | torch.device | None = None,
        torch_dtype: torch.dtype | None = None,
        load_on_init: bool = True,
    ) -> None:
        self.model_id = model_id
        self.torch_dtype = torch_dtype
        self.device = self._resolve_device(device)

        self.model: Any | None = None
        self.processor: Any | None = None

        if load_on_init:
            self.load_model()

    def load_model(self) -> None:
        model_kwargs = {}
        if self.torch_dtype is not None:
            model_kwargs["torch_dtype"] = self.torch_dtype

        self.model = AutoModel.from_pretrained(self.model_id, **model_kwargs)
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = self.model.to(self.device)

    @staticmethod
    def _resolve_device(device: str | torch.device | None) -> torch.device:
        if device is None:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    @property
    def is_loaded(self) -> bool:
        return self.model is not None and self.processor is not None

    @abstractmethod
    def explain(self, image: Any, text: str, **kwargs: Any) -> Any:
        """Return Grad-ECLIP explanation artifacts for an image-text pair."""

    def forward(self, image: Any, text: str, **processor_kwargs: Any) -> Any:
        if not self.is_loaded:
            raise RuntimeError("Model is not loaded. Call load_model() first.")

        kwargs = {"return_tensors": "pt", "padding": True}
        kwargs.update(processor_kwargs)

        inputs = self.processor(text=[text], images=image, **kwargs)
        inputs = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in inputs.items()}

        return self.model(**inputs)

    def print_model_info(self) -> None:
        if not self.is_loaded:
            print("Model is not loaded. Call load_model() first.")
            return

        print(f"Model ID: {self.model_id}")
        print(f"Device: {self.device}")
        print(f"Model type: {type(self.model).__name__}")
        print(f"Processor type: {type(self.processor).__name__}")
        print(f"Model architecture: {self.model}")

    def move_to_gpu(self, gpu_index: int = 0) -> None:
        """Move already loaded model to a chosen CUDA device."""
        if not self.is_loaded:
            raise RuntimeError("Model is not loaded. Call load_model() first.")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available on this machine.")

        self.device = torch.device(f"cuda:{gpu_index}")
        self.model = self.model.to(self.device)

    def offload_from_gpu(self) -> None:
        """Move model back to CPU and free cached CUDA memory."""
        if self.model is None:
            return

        self.device = torch.device("cpu")
        self.model = self.model.to(self.device)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def unload_model(self) -> None:
        """Fully unload model and processor from memory."""
        self.model = None
        self.processor = None
        self.device = torch.device("cpu")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# === GRAD-ECLIP IMPLEMENTATIONS ===
    @abstractmethod
    def encode_dense(self, pixel_values: torch.Tensor) -> EncodeDenseOutput:
        """
        Forward pass through vision encoder with decomposed last attention layer.
        
        Each subclass implements this with knowledge of its architecture
        (ViT-B/16 structure differs from SigLIP which differs from MetaCLIP, etc).
        
        Args:
            pixel_values: (batch, 3, H, W) preprocessed image tensor
            
        Returns:
            EncodeDenseOutput with q,k,v,attention from last layer needed for Grad-ECLIP.
        """
        pass

    # Exact copy from notebook
    # Used to compute att for explained layer 
    @staticmethod
    def _attention_layer(
        q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
        num_heads: int = 1, attn_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute scaled dot-product attention (universal for all models)."""
        tgt_len, bsz, embed_dim = q.shape
        head_dim = embed_dim // num_heads
        scaling = float(head_dim) ** -0.5
        q = q * scaling

        q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
        k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        attn_output_weights = torch.bmm(q, k.transpose(1, 2))

        if attn_mask is not None:
            attn_output_weights += attn_mask

        attn_output_weights = torch.nn.functional.softmax(attn_output_weights, dim=-1)
        attn_output_heads = torch.bmm(attn_output_weights, v)
        assert list(attn_output_heads.size()) == [bsz * num_heads, tgt_len, head_dim]

        attn_output = attn_output_heads.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, -1)
        attn_output_weights = attn_output_weights.sum(dim=1) / num_heads
        return attn_output, attn_output_weights

    # Exact copy from notebook 
    def _grad_eclip(
            self, c: torch.Tensor, q_out: torch.Tensor, k_out: torch.Tensor,
            v: torch.Tensor, att_output: torch.Tensor, patch_map_size: tuple,
            withksim: bool = True, cls_token: bool = True,
        ) -> torch.Tensor:
            """Compute Grad-ECLIP heatmap for scalar c (universal)."""
            grad = torch.autograd.grad(c, att_output, retain_graph=True, create_graph=True)[0]
            grad_cls = grad[:1, 0, :]

            if withksim:
                q_cls = q_out[:1, 0, :]
                k_patch = k_out[:, 0, :]
                q_cls = torch.nn.functional.normalize(q_cls, dim=-1)
                k_patch = torch.nn.functional.normalize(k_patch, dim=-1)
                cosine_qk = (q_cls * k_patch).sum(-1)
                cosine_qk = (cosine_qk - cosine_qk.min()) / (cosine_qk.max() - cosine_qk.min() + 1e-8)
                emap = torch.nn.functional.relu((grad_cls * v[:, 0, :] * cosine_qk[:, None]).sum(-1))
            else:
                emap = torch.nn.functional.relu((grad_cls * v[:, 0, :]).sum(-1))

            print(f"emap shape before reshape: {emap.shape}")
            if cls_token:
                emap = emap[1:]  # remove CLS token
            return emap.reshape(*patch_map_size)

    def _get_patch_size(self):
        return self.model.config.vision_config.patch_size
    
    def explain(self, image: Image.Image, text: str, keepsize: bool = True) -> dict[str, Any]:
        """
        Compute Grad-ECLIP explanation. Universal for all model subclasses.
        
        Args:
            image: PIL Image
            text: text caption
            
        Returns:
            dict with 'heatmap', 'logits', 'text_embedding', 'image_embedding'
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # 1. Preprocess image
        pixel_values: torch.Tensor = None
        if keepsize:
            pixel_values = self.proccess_keepsize(image).unsqueeze(0).to(self.device).detach()
        else:
            inputs = self.processor(images=image, return_tensors="pt")
            pixel_values = inputs['pixel_values'].to(self.device)

        # 2. Preprocess text
        text_inputs = self.processor(text=[text], return_tensors="pt").to(self.device)

        # 3. Get text embedding 
        text_features = self.model.get_text_features(
            **text_inputs
        ).pooler_output
        text_embedding = torch.nn.functional.normalize(text_features, dim=-1)

        # 4. Encode image (each subclass implements this differently)
        dense_output = self.encode_dense(pixel_values)

        # 5. Get image embedding from CLS token
        img_embedding = torch.nn.functional.normalize(dense_output.embeddings, dim=-1)

        # 6. Compute cosine similarities
        cosines = (img_embedding @ text_embedding.T)[0]
        

        # 7. Compute Grad-ECLIP for each scalar
        emap_list = [
            self._grad_eclip(c, dense_output.q_out, dense_output.k_out, dense_output.v,
                             dense_output.att_output, dense_output.patch_map_size, withksim=True,
                             cls_token=self.cls_token)
            for c in cosines
        ]
        emap = torch.stack(emap_list, dim=0).sum(0)

        # 8. Normalize heatmap
        emap = emap - emap.min()
        emap = emap / (emap.max() + 1e-8)

        return emap, dense_output.classic_output

    def _get_transform(self):
        image_processor = getattr(self.processor, "image_processor", self.processor)

        mean = tuple(float(x) for x in image_processor.image_mean)
        std = tuple(float(x) for x in image_processor.image_std)

        return torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean, std),
        ]) 

    def proccess_keepsize(self, img, scale_factor=1.0):
        w, h = img.size
        patch_height = patch_width = self._get_patch_size()

        # scale the image by scale_factor, and then round the sides to be divisible by patch length
        new_width = int(w * scale_factor / patch_width + 0.5) * patch_width
        new_height = int(h * scale_factor / patch_height + 0.5) * patch_height

        ResizeOp = torchvision.transforms.Resize((new_height, new_width), interpolation=torchvision.transforms.InterpolationMode.BICUBIC)
        img = ResizeOp(img).convert("RGB")
        return self._get_transform()(img)

    def patch_and_embed_with_interpolation(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # vision_width = model.config.vision_config.hidden_size
        # vision_heads = vision_width // 64

        clip_inres = self.model.config.vision_config.image_size
        clip_ksize = self.model.vision_model.embeddings.patch_embedding.kernel_size

        # modified from CLIP
        #x = x.half()


        x = self.model.vision_model.embeddings.patch_embedding(pixel_values)
        feah, feaw = x.shape[-2:]

        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        class_embedding = self.model.vision_model.embeddings.class_embedding.to(x.dtype)

        x = torch.cat([class_embedding + torch.zeros(x.shape[0], 1, x.shape[-1]).to(x), x], dim=1)

        #pos_embedding = clipmodel.visual.positional_embedding.to(x.dtype) HF
        pos_embedding = self.model.vision_model.embeddings.position_embedding.weight

        tok_pos, img_pos = pos_embedding[:1, :], pos_embedding[1:, :]
        pos_h = clip_inres // clip_ksize[0]
        pos_w = clip_inres // clip_ksize[1]
        assert img_pos.size(0) == (pos_h * pos_w), f"the size of pos_embedding ({img_pos.size(0)}) does not match resolution shape pos_h ({pos_h}) * pos_w ({pos_w})"
        img_pos = img_pos.reshape(1, pos_h, pos_w, img_pos.shape[1]).permute(0, 3, 1, 2)
        img_pos = torch.nn.functional.interpolate(img_pos, size=(feah, feaw), mode='bicubic', align_corners=False)
        img_pos = img_pos.reshape(1, img_pos.shape[1], -1).permute(0, 2, 1)
        pos_embedding = torch.cat((tok_pos[None, ...], img_pos), dim=1)

        return x + pos_embedding, (feah, feaw)

    def encode_dense(self, pixel_values: torch.Tensor) -> EncodeDenseOutput:
        # Potentially cound be deleted
        # x, patch_map_size = self.patch_and_embed_with_interpolation(pixel_values)

        y = self.model.vision_model.embeddings(pixel_values, interpolate_pos_encoding=True)
        patch_size = self._get_patch_size()
        patch_map_size = (pixel_values.shape[-2] // patch_size, pixel_values.shape[-1] // patch_size)
        print(f"y.shape: {y.shape}")
        print(patch_map_size)
        x = y

        if hasattr(self.model.vision_model, "pre_layrnorm"):
            x = self.model.vision_model.pre_layrnorm(x)

        x_in = x
        for module in self.model.vision_model.encoder.layers[:-1]:
            x_in = module(x_in, None)

        if hasattr(self.model, "visual_projection"):
            classic_prediction = self.model.visual_projection(self.model.vision_model.post_layernorm(self.model.vision_model.encoder.layers[-1](x_in, None)[:,0,:]))
        else:
            classic_prediction = self.model.vision_model.head(self.model.vision_model.post_layernorm(self.model.vision_model.encoder.layers[-1](x_in, None)))

        x_in = x_in.permute(1, 0, 2)

        ##################
        # LastTR.attention
        targetTR_2 = self.model.vision_model.encoder.layers[-1]

        x_before_attn = targetTR_2.layer_norm1(x_in)

        q = targetTR_2.self_attn.q_proj(x_before_attn)
        k = targetTR_2.self_attn.k_proj(x_before_attn)
        v = targetTR_2.self_attn.v_proj(x_before_attn)

        attn_output, attn = ClipModel._attention_layer(q, k, v, 1) #vision_heads

        x_after_attn = targetTR_2.self_attn.out_proj(attn_output)

        x = x_after_attn + x_in

        # x_out = x + targetTR.mlp(targetTR.ln_2(x))
        x_out = x + targetTR_2.mlp(targetTR_2.layer_norm2(x))

        x = x_out.permute(1, 0, 2)  # LND -> NLD

        #x = clipmodel.visual.ln_post(x) HF
        x = self.model.vision_model.post_layernorm(x)

        # x = x @ clipmodel.visual.proj HF

        if hasattr(self.model, "visual_projection"):
            x = self.model.visual_projection(x)[:, 0, :]
        else:
            x = self.model.vision_model.head(x)

        ## ==== get lastv ==============
        qkv = torch.stack((q, k, v), dim=0)
        qkv = targetTR_2.self_attn.out_proj(qkv)
        q_out, k_out, v_out = qkv[0], qkv[1], qkv[2]


        # TODO czy my tego używamy
        # v_final = v_out + x_in
        # v_final = v_final + targetTR_2.mlp(targetTR_2.layer_norm2(v_final))
        # v_final = v_final.permute(1, 0, 2)

        # v_final = self.model.vision_model.post_layernorm(v_final)

        # v_final = self.model.visual_projection(v_final)
        ##############

        print("embeddings shape:", x.shape)
        return EncodeDenseOutput(
            embeddings=x,
            q_out=q_out,
            k_out=k_out,
            v=v,
            att_output=attn_output,
            patch_map_size=patch_map_size,
            classic_output=classic_prediction
        )
