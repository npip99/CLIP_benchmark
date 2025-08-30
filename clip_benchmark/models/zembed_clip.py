from typing import Dict, cast, Any
import numpy as np
import torch
import torch.nn.functional as F
from PIL.Image import Image, Resampling

from transformers.models.auto.modeling_auto import AutoModel
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VisionTransformerPretrainedModel,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLPatchMerger,
)
from transformers.models.qwen2_5_vl.modular_qwen2_5_vl import Qwen2_5_VLProcessor
from transformers.models.qwen3.modeling_qwen3 import Qwen3Model
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast


def get_model(
    rank: int,
    device: torch.device,
) -> tuple[
    Qwen3Model,
    PreTrainedTokenizerFast,
    Qwen2_5_VisionTransformerPretrainedModel,
    Qwen2_5_VLProcessor,
]:
    """Load all models and setup training configuration"""
    multimodal = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-3B-Instruct"
    )
    processor = Qwen2_5_VLProcessor.from_pretrained(
        "Qwen/Qwen2.5-VL-3B-Instruct", use_fast=True
    )
    assert isinstance(processor, Qwen2_5_VLProcessor)

    # Extract components
    visual_encoder = multimodal.visual
    embedding_model = AutoModel.from_pretrained("Qwen/Qwen3-Embedding-4B")
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen3-Embedding-4B", padding_side="right"
    )

    # Move to device
    visual_encoder = visual_encoder.to(device)  # pyright: ignore[reportArgumentType]
    embedding_model = embedding_model.to(device)

    # Add final projection to match dimensions
    # Get dimensions from merger config
    vision_dim = visual_encoder.merger.mlp[
        -1
    ].out_features  # Output dimension from final linear layer
    text_dim = embedding_model.config.hidden_size  # Text embedding dimension

    if vision_dim != text_dim:
        visual_encoder.config.out_hidden_size = text_dim
        visual_encoder.merger = Qwen2_5_VLPatchMerger(
            dim=visual_encoder.config.out_hidden_size,  # pyright: ignore[reportUnknownArgumentType]
            context_dim=visual_encoder.config.hidden_size,  # pyright: ignore[reportUnknownArgumentType]
            spatial_merge_size=visual_encoder.config.spatial_merge_size,  # pyright: ignore[reportUnknownArgumentType]
        ).to(device)

    # Freeze all parameters in embedding model
    embedding_model.eval()
    for param in embedding_model.parameters():
        param.requires_grad = False

    # Freeze vision transformer except merger
    visual_encoder.eval()
    # visual_encoder.gradient_checkpointing_enable()
    for name, param in visual_encoder.named_parameters():
        if "merger" not in name:
            param.requires_grad = False

    if rank == 0:
        # Print parameter counts
        print("=" * 60)
        print("PARAMETER COUNTS")
        print("=" * 60)

        visual_total = sum(p.numel() for p in visual_encoder.parameters())
        visual_trainable = sum(
            p.numel() for p in visual_encoder.parameters() if p.requires_grad
        )
        emb_total = sum(p.numel() for p in cast(Any, embedding_model.parameters()))
        merger_params = sum(
            p.numel() for p in visual_encoder.merger.parameters() if p.requires_grad
        )

        print(
            f"Visual Encoder: {visual_total:,} total ({visual_trainable:,} trainable)"
        )
        print(f"Text Embedding: {emb_total:,} total (frozen)")
        print(f"Merger only: {merger_params:,} trainable")
        print(
            f"Combined: {visual_total + emb_total:,} total ({visual_trainable:,} trainable)"
        )
        print("=" * 60)

    return embedding_model, tokenizer, visual_encoder, processor


class ZembedModel:
    def __init__(self, path: str, device: torch.device):
        embedding_model, tokenizer, visual_encoder, processor = get_model(0, device)

        merger_state_dict = torch.load(
            "/home/user/ml/data/checkpoints/pretrained_merger_hard_negatives/epoch-001-step-54250/model.pth"
        )
        visual_encoder.merger.load_state_dict(merger_state_dict)

        self.embedding_model = embedding_model
        self.tokenizer = tokenizer
        self.visual_encoder = visual_encoder
        self.processor = processor
        self.device = device

    def eval(self):
        pass

    def encode_text(self, tokens):
        texts = tokens.texts
        batch_inputs = self.tokenizer(
            texts,
            truncation=True,
            max_length=4096,
            padding=True,
            return_tensors="pt",
        )
        batch_inputs = batch_inputs.to(self.device)

        with torch.no_grad():
            outputs = self.embedding_model(**batch_inputs, use_cache=False)

        last_hidden_states = cast(torch.Tensor, outputs.last_hidden_state)
        attention_mask = cast(torch.Tensor, batch_inputs.attention_mask)
        last_positions = attention_mask.sum(dim=1) - 1

        batch_size = last_hidden_states.shape[0]
        batch_indices = torch.arange(batch_size, device=self.device)
        batch_embeddings = last_hidden_states[batch_indices, last_positions]
        batch_embeddings = F.normalize(batch_embeddings, p=2, dim=1)
        return batch_embeddings

    def encode_image(self, images):
        device = self.device

        inputs = self.processor(
            text=["<|image_pad|>"] * len(images),
            images=images,
            padding=True,
            return_tensors="pt",
        )
        pixel_values = inputs["pixel_values"].to(device)
        image_grid_thw = inputs["image_grid_thw"].to(device)

        batch_size = image_grid_thw.shape[0]
        seq_lengths = [
            int(seq_len)
            for seq_len in image_grid_thw[:, 0]
            * (image_grid_thw[:, 1] // self.visual_encoder.spatial_merge_size)
            * (image_grid_thw[:, 2] // self.visual_encoder.spatial_merge_size)
        ]

        vision_embeddings = self.visual_encoder(pixel_values, grid_thw=image_grid_thw)

        # Find max sequence length for padding
        max_seq_len = max(seq_lengths) + 1  # +1 for the end token

        # Get the embedding for the pad token (<|endoftext|>)
        assert isinstance(self.tokenizer.pad_token_id, int)
        pad_token_tensor = torch.tensor([self.tokenizer.pad_token_id], device=device)
        end_token_embedding = self.embedding_model.embed_tokens(
            pad_token_tensor
        )  # Shape: [1, hidden_dim]
        end_token_embedding = end_token_embedding.squeeze(0)  # Shape: [hidden_dim]

        # Create padded sequences with end tokens
        hidden_dim = vision_embeddings.shape[-1]
        inputs_embeds = torch.zeros(batch_size, max_seq_len, hidden_dim, device=device)
        attention_mask = torch.zeros(
            batch_size, max_seq_len, dtype=torch.int64, device=device
        )

        start_idx = 0
        for i in range(batch_size):
            seq_len = seq_lengths[i]
            end_idx = start_idx + seq_len

            # Copy vision embeddings
            inputs_embeds[i, :seq_len] = vision_embeddings[start_idx:end_idx]

            # Add end token embedding
            inputs_embeds[i, seq_len] = end_token_embedding

            # Set attention mask (1 for real tokens, 0 for padding)
            attention_mask[i, : seq_len + 1] = 1

            start_idx = end_idx

        outputs = self.embedding_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            use_cache=False,
        )

        last_hidden_states = cast(torch.Tensor, outputs.last_hidden_state)
        last_positions = attention_mask.sum(dim=1) - 1  # Position of end token

        batch_indices = torch.arange(batch_size, device=device)
        batch_embeddings = last_hidden_states[batch_indices, last_positions]
        batch_embeddings = F.normalize(batch_embeddings, p=2, dim=1)

        return batch_embeddings


class ZembedTokens:
    def __init__(self, texts: list[str]) -> None:
        self.texts = texts

    def to(self, device) -> "ZembedTokens":
        return self


class ZembedTokenizer:
    def __init__(self) -> None:
        pass

    def __call__(self, texts) -> Any:
        return ZembedTokens(texts)


class ZembedTransform:
    def __init__(self, size: int = 224) -> None:
        self.size = size

    def __call__(self, image: Image) -> Any:
        return np.array(image.resize((self.size, self.size), Resampling.BICUBIC))


def load_zembed_clip(pretrained: str, device="cpu", **kwargs):
    device = torch.device(device)
    return ZembedModel(pretrained, device), ZembedTransform(), ZembedTokenizer()
