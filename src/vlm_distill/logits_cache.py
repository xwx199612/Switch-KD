from __future__ import annotations

from typing import Any


def compact_logits(logits, max_vocab: int | None) -> dict[str, Any] | list[list[list[float]]]:
    import torch

    logits = logits.detach().float().cpu()
    if not max_vocab:
        return logits.tolist()
    top_values, top_indices = torch.topk(logits, k=min(max_vocab, logits.shape[-1]), dim=-1)
    return {
        "indices": top_indices.tolist(),
        "values": top_values.tolist(),
        "shape": list(logits.shape),
        "vocab_size": int(logits.shape[-1]),
    }


def is_compact_logits(cached: Any) -> bool:
    return isinstance(cached, dict) and "indices" in cached and "values" in cached


def cached_vocab_size(cached: Any) -> int | None:
    if is_compact_logits(cached):
        if "vocab_size" in cached:
            return int(cached["vocab_size"])
        shape = cached.get("shape")
        if shape:
            return int(shape[-1])
    if isinstance(cached, list) and cached and isinstance(cached[0], list):
        return len(cached[0][0])
    return None


def materialize_cached_logits(cached: Any, *, device, dtype, vocab_size: int | None = None):
    import torch

    if is_compact_logits(cached):
        shape = tuple(cached["shape"])
        if vocab_size is None:
            vocab_size = cached_vocab_size(cached)
        if vocab_size is None:
            vocab_size = int(max(cached["indices"][-1][-1])) + 1
        tensor = torch.full(shape, torch.finfo(dtype).min, device=device, dtype=dtype)
        indices = torch.tensor(cached["indices"], device=device)
        values = torch.tensor(cached["values"], device=device, dtype=dtype)
        tensor.scatter_(-1, indices, values)
        return tensor

    tensor = torch.tensor(cached, device=device, dtype=dtype)
    if vocab_size is not None and tensor.shape[-1] != vocab_size:
        tensor = align_reference_logits(tensor, target_shape=(*tensor.shape[:-1], vocab_size), dtype=dtype)
    return tensor


def align_reference_logits(reference, *, target_shape: tuple[int, ...], dtype=None):
    """Pad or truncate reference logits to match student logits shape."""
    import torch

    if len(target_shape) != 3:
        raise ValueError(f"Expected target_shape (batch, seq, vocab), got {target_shape}")

    fill_value = torch.finfo(dtype or reference.dtype).min
    batch_size, seq_len, vocab_size = target_shape
    aligned = reference

    if aligned.shape[0] != batch_size:
        if aligned.shape[0] == 1 and batch_size > 1:
            aligned = aligned.expand(batch_size, -1, -1)
        else:
            aligned = aligned[:batch_size]

    if aligned.shape[-1] < vocab_size:
        pad = torch.full(
            (aligned.shape[0], aligned.shape[1], vocab_size - aligned.shape[-1]),
            fill_value,
            device=aligned.device,
            dtype=aligned.dtype,
        )
        aligned = torch.cat([aligned, pad], dim=-1)
    elif aligned.shape[-1] > vocab_size:
        aligned = aligned[..., :vocab_size]

    if aligned.shape[1] < seq_len:
        pad = torch.full(
            (aligned.shape[0], seq_len - aligned.shape[1], vocab_size),
            fill_value,
            device=aligned.device,
            dtype=aligned.dtype,
        )
        aligned = torch.cat([aligned, pad], dim=1)
    elif aligned.shape[1] > seq_len:
        aligned = aligned[:, :seq_len, :]

    return aligned


def align_reference_logits_to_suffix(
    reference,
    *,
    target_shape: tuple[int, ...],
    reference_prompt_len: int | None,
    student_prompt_len: int | None,
    dtype=None,
):
    """Align cached logits by matching answer suffixes when prompt lengths differ."""
    import torch

    if reference_prompt_len is None or student_prompt_len is None:
        return align_reference_logits(reference, target_shape=target_shape, dtype=dtype)

    ref_answer = reference[:, int(reference_prompt_len) :, :]
    batch_size, seq_len, vocab_size = target_shape
    answer_len = seq_len - int(student_prompt_len)
    if answer_len <= 0:
        return align_reference_logits(reference, target_shape=target_shape, dtype=dtype)

    fill_value = torch.finfo(dtype or reference.dtype).min
    if ref_answer.shape[1] > answer_len:
        ref_answer = ref_answer[:, :answer_len, :]
    elif ref_answer.shape[1] < answer_len:
        pad = torch.full(
            (ref_answer.shape[0], answer_len - ref_answer.shape[1], ref_answer.shape[-1]),
            fill_value,
            device=ref_answer.device,
            dtype=ref_answer.dtype,
        )
        ref_answer = torch.cat([ref_answer, pad], dim=1)

    aligned = torch.full(
        (batch_size, seq_len, ref_answer.shape[-1]),
        fill_value,
        device=reference.device,
        dtype=reference.dtype,
    )
    aligned[:, int(student_prompt_len) : int(student_prompt_len) + ref_answer.shape[1], :] = ref_answer
    return align_reference_logits(aligned, target_shape=target_shape, dtype=dtype)


def vocab_sizes_compatible(reference_vocab: int | None, student_vocab: int) -> bool:
    return reference_vocab is not None and reference_vocab == student_vocab
