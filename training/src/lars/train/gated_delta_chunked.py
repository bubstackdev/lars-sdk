"""A chunked, differentiable gated delta rule for training Qwen3.5's Gated DeltaNet layers in MLX.

mlx_lm's Metal kernel for the gated delta rule has no gradient, so in training mode it falls back
to a token-by-token loop that keeps a full recurrent state per token per layer: gigabytes per
layer for one 4,096-token batch, and far too slow. This is the standard chunkwise form (as in
flash-linear-attention's reference `chunk_gated_delta_rule`): within a chunk of C tokens the
recurrence is a few matrix products after one blocked triangular inverse (the UT transform), and
only one state per chunk is carried. Everything is plain MLX ops, so autodiff works.

The recurrence, per head, with the state S of shape [Dv, Dk] as mlx_lm keeps it:
    S_t = g_t S_{t-1} (I - beta_t k_t k_t^T) + beta_t v_t k_t^T,    y_t = S_t q_t
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import mlx.nn as nn

CHUNK = 64


def log_decay(A_log: mx.array, a: mx.array, dt_bias: mx.array) -> mx.array:
    """log g, without the underflow of taking the log of mlx_lm's compute_g."""
    return -mx.exp(A_log.astype(mx.float32)) * nn.softplus((a + dt_bias).astype(mx.float32))


def _unit_lower_inverse(A: mx.array, leaf: int = 16) -> mx.array:
    """(I + A)^-1 for strictly lower-triangular A [..., C, C]: forward substitution on the
    diagonal leaf blocks (all at once), then exact 2x2 block combination,
    [[T11, 0], [T21, T22]] with T21 = -T22 A21 T11, up to the full size."""
    C = A.shape[-1]
    leaf = min(leaf, C)
    n = C // leaf
    blocks = mx.stack([A[..., i * leaf:(i + 1) * leaf, i * leaf:(i + 1) * leaf] for i in range(n)], axis=-3)
    eye = mx.eye(leaf, dtype=A.dtype)
    rows = [mx.broadcast_to(eye[0], blocks.shape[:-2] + (leaf,))]
    for i in range(1, leaf):
        done = mx.stack(rows, axis=-2)  # [..., n, i, leaf]
        rows.append(eye[i] - (blocks[..., i, :i, None] * done).sum(axis=-2))
    inv = mx.stack(rows, axis=-2)  # [..., n, leaf, leaf]
    size = leaf
    while size < C:
        pairs = []
        for p in range(inv.shape[-3] // 2):
            t11, t22 = inv[..., 2 * p, :, :], inv[..., 2 * p + 1, :, :]
            r0, c0 = (2 * p + 1) * size, 2 * p * size
            a21 = A[..., r0:r0 + size, c0:c0 + size]
            t21 = -(t22 @ (a21 @ t11))
            top = mx.concatenate([t11, mx.zeros_like(t11)], axis=-1)
            bottom = mx.concatenate([t21, t22], axis=-1)
            pairs.append(mx.concatenate([top, bottom], axis=-2))
        inv = mx.stack(pairs, axis=-3)
        size *= 2
    return inv[..., 0, :, :]


def gated_delta_chunked(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    log_g: mx.array,
    beta: mx.array,
    state: mx.array | None = None,
    chunk: int = CHUNK,
) -> tuple[mx.array, mx.array]:
    """q, k: [B, T, Hk, Dk]; v: [B, T, Hv, Dv]; log_g, beta: [B, T, Hv]; state: [B, Hv, Dv, Dk].

    Returns y [B, T, Hv, Dv] (in q's dtype) and the final state [B, Hv, Dv, Dk] (float32)."""
    B, T, Hk, Dk = q.shape
    Hv, Dv = v.shape[-2:]
    out_dtype = q.dtype
    if (repeat := Hv // Hk) > 1:
        q = mx.repeat(q, repeat, -2)
        k = mx.repeat(k, repeat, -2)
    pad = (-T) % chunk
    if pad:
        # Padding with beta = 0 and log g = 0 leaves the state untouched.
        widths = [(0, 0), (0, pad), (0, 0), (0, 0)]
        q, k, v = (mx.pad(x, widths) for x in (q, k, v))
        log_g, beta = (mx.pad(x, [(0, 0), (0, pad), (0, 0)]) for x in (log_g, beta))
    n = (T + pad) // chunk

    def chunks(x: mx.array) -> mx.array:  # [B, T, H, ...] -> [B, H, N, C, ...]
        x = x.astype(mx.float32).reshape(B, n, chunk, Hv, *x.shape[3:])
        return mx.moveaxis(x, 3, 1)

    q, k, v, log_g, beta = (chunks(x) for x in (q, k, v, log_g, beta))
    G = mx.cumsum(log_g, axis=-1)  # [B, H, N, C]
    lower = mx.tril(mx.ones((chunk, chunk), dtype=mx.bool_))
    # decay[i, j] = exp(G_i - G_j) for j <= i, else 0
    decay = mx.where(lower, mx.exp(mx.where(lower, G[..., :, None] - G[..., None, :], 0.0)), 0.0)
    k_beta = k * beta[..., None]
    strictly = mx.tril(mx.ones((chunk, chunk), dtype=mx.bool_), k=-1)
    A = mx.where(strictly, (k_beta @ mx.swapaxes(k, -1, -2)) * decay, 0.0)
    # The UT transform, (I + A)^-1, inverted blockwise: exact and stable, unlike a power series
    # in A, and 3 to 5 times faster than forward substitution over all 64 rows.
    inverse = _unit_lower_inverse(A)
    u = inverse @ (v * beta[..., None])  # [B, H, N, C, Dv]
    w = inverse @ (k_beta * mx.exp(G)[..., None])  # [B, H, N, C, Dk]
    attn = (q @ mx.swapaxes(k, -1, -2)) * decay  # causal, diagonal included
    q_decayed = q * mx.exp(G)[..., None]
    k_to_end = k * mx.exp(G[..., -1:] - G)[..., None]

    S = mx.zeros((B, Hv, Dk, Dv), mx.float32) if state is None else mx.swapaxes(state.astype(mx.float32), -1, -2)
    outputs = []
    for i in range(n):
        v_new = u[:, :, i] - w[:, :, i] @ S  # [B, H, C, Dv]
        outputs.append(q_decayed[:, :, i] @ S + attn[:, :, i] @ v_new)
        S = S * mx.exp(G[:, :, i, -1])[..., None, None] + mx.swapaxes(k_to_end[:, :, i], -1, -2) @ v_new
    y = mx.stack(outputs, axis=2)  # [B, H, N, C, Dv]
    y = mx.moveaxis(y, 1, 3).reshape(B, n * chunk, Hv, Dv)[:, :T]
    return y.astype(out_dtype), mx.swapaxes(S, -1, -2)


def use_for_training() -> bool:
    """Route mlx_lm's Qwen3.5 layers to the chunked rule whenever they would take the slow path
    (training mode, no padding mask). Returns False when this mlx_lm has no Qwen3.5 model."""
    try:
        from mlx_lm.models import qwen3_5
    except ImportError:
        return False
    original = qwen3_5.gated_delta_update

    def update(q, k, v, a, b, A_log, dt_bias, state=None, mask=None, use_kernel=True, **kwargs: Any):
        if use_kernel or mask is not None:
            return original(q, k, v, a, b, A_log, dt_bias, state, mask, use_kernel=use_kernel, **kwargs)
        return gated_delta_chunked(q, k, v, log_decay(A_log, a, dt_bias), mx.sigmoid(b), state)

    qwen3_5.gated_delta_update = update
    return True
