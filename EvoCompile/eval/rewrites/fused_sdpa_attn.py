"""Swap unfused bmm+softmax attention for fused SDPA (in-tree or Triton).

Hooks:
  DistilBERT MultiHeadSelfAttention
  BERT_pytorch Attention (already-projected QKV)
  HF BertSelfAttention (absolute positions)
  OPT/XGLM-style q_proj/k_proj/v_proj/out_proj decoder attn (causal)
  T5Attention / MT5Attention (q/k/v/o + 4D relative bias)
  LongformerSelfAttention (sliding-window; skip global-attn branch)

Graph-break / Dynamo notes:
  Do not getattr through nn.Module.__getattr__ (is_decoder).
  Do not data-dependent Python branches (count_nonzero, tensor.item).
  Store per-module flags at apply() time.
  Raise cache_size_limit so per-layer MethodType frames still compile.
"""

from __future__ import annotations

import types

import torch
import torch.nn.functional as F

# Bound at apply() from kernels.attn_call() — this recipe's torch.ops.gko.*.
# Must be a module global so Dynamo LOAD_GLOBAL works (no import inside _call).
_ATTN_CALL = None


def _bump_dynamo_cache():
    try:
        import torch._dynamo as dynamo

        dynamo.config.cache_size_limit = max(int(dynamo.config.cache_size_limit), 256)
    except Exception:
        pass


def _shape(x, bs, n_heads, dim_per_head):
    return x.view(bs, -1, n_heads, dim_per_head).transpose(1, 2)


def _unshape(x, bs, dim):
    return x.transpose(1, 2).contiguous().view(bs, -1, dim)


def _call(q, k, v, bias, scale, causal, use_in_tree, window=0):
    if use_in_tree:
        if causal and bias is None:
            return F.scaled_dot_product_attention(
                q, k, v, dropout_p=0.0, scale=scale, is_causal=True
            )
        mask = None
        if bias is not None:
            mask = bias.to(dtype=q.dtype)
            while mask.dim() < 4:
                mask = mask.unsqueeze(1)
        return F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=0.0, scale=scale, is_causal=False
        )
    fn = _ATTN_CALL
    if fn is None:
        raise RuntimeError(
            "no torch.ops.gko attention bound; replace_pattern must register "
            "the generated op (or catalog fused_sdpa) before fused_sdpa_attn"
        )
    return fn(q, k, v, bias, float(scale), int(causal), window=int(window or 0))


def _bias_keep_mask(mask, bs, k_len, device, dtype):
    if mask is None:
        return None
    m = mask
    if m.dtype == torch.bool:
        keep = m
    else:
        keep = m != 0
    keep = keep.reshape(bs, -1)[:, :k_len]
    bias = torch.zeros(bs, k_len, device=device, dtype=dtype)
    return bias.masked_fill(~keep, torch.finfo(dtype).min)


def _bias_additive(mask, bs, k_len, dtype):
    if mask is None:
        return None
    m = mask
    if m.dim() == 4:
        m = m[:, 0, -1, :]
    elif m.dim() == 3:
        m = m[:, -1, :]
    return m.reshape(bs, -1)[:, :k_len].to(dtype)


def _mha_forward(self, query, key, value, mask, head_mask=None, output_attentions=False, *, _tree=False):
    q = self.q_lin(query)
    k = self.k_lin(key)
    v = self.v_lin(value)
    bs = q.size(0)
    dim = self.dim
    n_heads = self.n_heads
    dim_per_head = dim // n_heads
    qh = _shape(q, bs, n_heads, dim_per_head).contiguous()
    kh = _shape(k, bs, n_heads, dim_per_head).contiguous()
    vh = _shape(v, bs, n_heads, dim_per_head).contiguous()
    scale = 1.0 / (dim_per_head ** 0.5)
    bias = _bias_keep_mask(mask, bs, kh.size(2), qh.device, qh.dtype)
    ctx = _call(qh, kh, vh, bias, scale, 0, _tree)
    if head_mask is not None:
        ctx = ctx * head_mask
    out = self.out_lin(_unshape(ctx, bs, dim))
    return out, None


def _bert_pytorch_attn(self, query, key, value, dropout, mask=None, *, _tree=False):
    scale = 1.0 / (query.size(-1) ** 0.5)
    bs = query.size(0)
    k_len = key.size(-2)
    bias = _bias_keep_mask(mask, bs, k_len, query.device, query.dtype)
    ctx = _call(query, key, value, bias, scale, 0, _tree)
    return ctx, None


def _bert_self_forward(
    self,
    hidden_states,
    attention_mask=None,
    head_mask=None,
    encoder_hidden_states=None,
    encoder_attention_mask=None,
    past_key_value=None,
    output_attentions=False,
    *,
    _tree=False,
    _pos_type="absolute",
    _tuple_out=True,
    _is_decoder=False,
):
    if encoder_hidden_states is not None or past_key_value is not None:
        return self._gko_orig_forward(
            hidden_states,
            attention_mask=attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
        )
    if _pos_type not in ("absolute", None):
        return self._gko_orig_forward(
            hidden_states,
            attention_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
        )
    bs = hidden_states.size(0)
    qh = self.transpose_for_scores(self.query(hidden_states))
    kh = self.transpose_for_scores(self.key(hidden_states))
    vh = self.transpose_for_scores(self.value(hidden_states))
    scale = 1.0 / (self.attention_head_size ** 0.5)
    bias = _bias_additive(attention_mask, bs, kh.size(2), qh.dtype)
    ctx = _call(qh, kh, vh, bias, scale, 0, _tree)
    if head_mask is not None:
        ctx = ctx * head_mask
    context = ctx.permute(0, 2, 1, 3).contiguous()
    context = context.view(context.size()[:-2] + (self.all_head_size,))
    if not _tuple_out:
        return context
    outputs = (context, None) if output_attentions else (context,)
    if _is_decoder:
        outputs = outputs + ((kh, vh),)
    return outputs


def _opt_like_forward(
    self,
    hidden_states,
    key_value_states=None,
    past_key_value=None,
    attention_mask=None,
    layer_head_mask=None,
    output_attentions=False,
    *,
    _tree=False,
    _causal=True,
    _scale_q=False,
):
    if key_value_states is not None or past_key_value is not None:
        return self._gko_orig_forward(
            hidden_states,
            key_value_states=key_value_states,
            past_key_value=past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
    bsz, tgt_len, _ = hidden_states.size()
    q = self.q_proj(hidden_states)
    if _scale_q:
        q = q * self.scaling
        scale = 1.0
    else:
        scale = 1.0 / (self.head_dim ** 0.5)
    k = self.k_proj(hidden_states)
    v = self.v_proj(hidden_states)
    qh = self._shape(q, tgt_len, bsz)
    kh = self._shape(k, tgt_len, bsz)
    vh = self._shape(v, tgt_len, bsz)
    bias = None if _causal else _bias_additive(attention_mask, bsz, kh.size(2), qh.dtype)
    ctx = _call(qh, kh, vh, bias, scale, 1 if _causal else 0, _tree)
    if layer_head_mask is not None:
        ctx = ctx * layer_head_mask.view(1, -1, 1, 1)
    out = ctx.transpose(1, 2).reshape(bsz, tgt_len, self.embed_dim)
    out = self.out_proj(out)
    return out, None, None


def _t5_forward(
    self,
    hidden_states,
    mask=None,
    key_value_states=None,
    position_bias=None,
    past_key_value=None,
    layer_head_mask=None,
    query_length=None,
    use_cache=False,
    output_attentions=False,
    *,
    _tree=False,
    _is_decoder=False,
):
    if past_key_value is not None:
        return self._gko_orig_forward(
            hidden_states,
            mask=mask,
            key_value_states=key_value_states,
            position_bias=position_bias,
            past_key_value=past_key_value,
            layer_head_mask=layer_head_mask,
            query_length=query_length,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
    batch_size, seq_length = hidden_states.shape[:2]
    real_seq_length = seq_length
    key_length = real_seq_length if key_value_states is None else key_value_states.shape[1]

    query_states = _shape(self.q(hidden_states), batch_size, self.n_heads, self.key_value_proj_dim)
    src = hidden_states if key_value_states is None else key_value_states
    key_states = _shape(self.k(src), batch_size, self.n_heads, self.key_value_proj_dim)
    value_states = _shape(self.v(src), batch_size, self.n_heads, self.key_value_proj_dim)
    if position_bias is None:
        if not self.has_relative_attention_bias:
            position_bias = torch.zeros(
                (1, self.n_heads, real_seq_length, key_length),
                device=query_states.device,
                dtype=query_states.dtype,
            )
        else:
            position_bias = self.compute_bias(
                real_seq_length, key_length, device=query_states.device
            )
        if mask is not None:
            position_bias = position_bias + mask
    if self.pruned_heads:
        keep = torch.ones(
            position_bias.shape[1], device=position_bias.device, dtype=torch.bool
        )
        keep[list(self.pruned_heads)] = False
        position_bias = position_bias[:, keep]
    # T5 scales inside q weights; 4D rel-bias is the mask. Decoder causal is in bias.
    ctx = _call(query_states, key_states, value_states, position_bias, 1.0, 0, _tree)
    if layer_head_mask is not None:
        ctx = ctx * layer_head_mask.view(1, -1, 1, 1)
    attn_output = self.o(_unshape(ctx, batch_size, self.inner_dim))
    present = (key_states, value_states) if (_is_decoder and use_cache) else None
    outputs = (attn_output,) + (present,) + (position_bias,)
    if output_attentions:
        outputs = outputs + (None,)
    return outputs


def _longformer_forward(
    self,
    hidden_states,
    attention_mask=None,
    layer_head_mask=None,
    is_index_masked=None,
    is_index_global_attn=None,
    is_global_attn=None,
    output_attentions=False,
    *,
    _tree=False,
    _window=0,
):
    if is_global_attn:
        return self._gko_orig_forward(
            hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            is_index_masked=is_index_masked,
            is_index_global_attn=is_index_global_attn,
            is_global_attn=is_global_attn,
            output_attentions=output_attentions,
        )
    batch_size, seq_len, embed_dim = hidden_states.size()

    def _heads(t):
        return t.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    qh = _heads(self.query(hidden_states))
    kh = _heads(self.key(hidden_states))
    vh = _heads(self.value(hidden_states))
    scale = 1.0 / (self.head_dim ** 0.5)
    bias = None
    mask_idx = is_index_masked
    if mask_idx is None and attention_mask is not None:
        mask_idx = attention_mask < 0
    if mask_idx is not None:
        keep = ~mask_idx
        if keep.dim() > 2:
            keep = keep.reshape(keep.size(0), -1)
        if keep.size(0) == batch_size and keep.size(-1) >= seq_len:
            bias = qh.new_zeros((batch_size, seq_len))
            bias = bias.masked_fill(~keep[:, :seq_len], torch.finfo(qh.dtype).min)
    ctx = _call(qh, kh, vh, bias, scale, 0, _tree, window=_window)
    if layer_head_mask is not None:
        ctx = ctx * layer_head_mask.view(1, -1, 1, 1)
    ctx = ctx.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
    outputs = (ctx,)
    if output_attentions:
        outputs = outputs + (None,)
    return outputs


def _bind(fn, tree, **flags):
    # Bind flags as defaults, not a nested closure over **flags — Dynamo
    # LOAD_GLOBAL AssertionError on import_source when inlining a closure
    # that later does a lazy import.
    def bound(self, *a, **k):
        if "_tree" not in k:
            k["_tree"] = tree
        for key, val in flags.items():
            k.setdefault(key, val)
        return fn(self, *a, **k)

    bound.__name__ = fn.__name__
    bound.__qualname__ = fn.__qualname__
    return bound


def _flag_module(m):
    is_dec = False
    try:
        is_dec = bool(object.__getattribute__(m, "is_decoder"))
    except AttributeError:
        is_dec = False
    object.__setattr__(m, "_gko_is_decoder", is_dec)
    pos = "absolute"
    try:
        pos = object.__getattribute__(m, "position_embedding_type")
    except AttributeError:
        pos = "absolute"
    object.__setattr__(m, "_gko_pos_type", pos)
    scale_q = False
    try:
        scale_q = bool(object.__getattribute__(m, "scaling"))
    except AttributeError:
        scale_q = False
    object.__setattr__(m, "_gko_scale_q", scale_q)
    causal = True
    try:
        causal = bool(object.__getattribute__(m, "is_causal"))
    except AttributeError:
        causal = True
    object.__setattr__(m, "_gko_causal", causal)
    tuple_out = "fastNLP" not in ((type(m).__module__) or "")
    object.__setattr__(m, "_gko_tuple_out", tuple_out)


def _infer_kind_and_layers(model):
    n_layer = 0
    cfg = getattr(model, "config", None)
    if cfg is not None:
        n_layer = int(
            getattr(cfg, "num_hidden_layers", 0)
            or getattr(cfg, "n_layer", 0)
            or getattr(cfg, "num_layers", 0)
            or 0
        )
    names = [type(model).__name__, type(model).__module__ or ""]
    for i, m in enumerate(model.modules()):
        if i > 80:
            break
        names.append(type(m).__name__)
    blob = " ".join(names).lower()
    if "longformer" in blob:
        return "longformer", n_layer
    if "t5" in blob or "mt5" in blob:
        return "t5_rel", n_layer
    if "xlnet" in blob:
        return "xlnet", n_layer
    if "swin" in blob:
        return "swin_window", n_layer
    if "gemma" in blob:
        return "gemma_rope", n_layer
    if any(k in blob for k in ("optattention", "optfor", "gptneo", "llama", "xglm", "causal")):
        return "causal", n_layer
    return "", n_layer


def apply(model, *, use_in_tree_sdpa: bool = False) -> str:
    import os

    _bump_dynamo_cache()
    tree = bool(use_in_tree_sdpa)
    if not tree:
        global _ATTN_CALL
        bound = None
        register_kernel = None
        try:
            from kernels import attn_call, register as register_kernel

            bound = attn_call()
        except Exception:
            bound = None
        # Do not overwrite a generated gko op with catalog fused_attention.
        if bound is None and register_kernel is not None:
            try:
                register_kernel("fused_sdpa", backend="auto")
            except Exception:
                pass
            try:
                from kernels import attn_call as _attn_call

                bound = _attn_call()
            except Exception:
                bound = None
            try:
                os.environ.setdefault("GKO_ATTN_BACKEND", "auto")
                from kernels.attn_policy import set_context

                kind, n_layer = _infer_kind_and_layers(model)
                set_context(n_layer=n_layer, kind=kind)
            except Exception:
                pass
        _ATTN_CALL = bound
    n = 0
    kinds = []
    for m in model.modules():
        if getattr(m, "_gko_fused_sdpa", False) or getattr(m, "_gko_s6_28", False) or getattr(m, "_gko_s103", False):
            continue
        name = type(m).__name__
        mod = type(m).__module__ or ""
        fn = None
        if name in ("T5Attention", "MT5Attention") and hasattr(m, "q") and hasattr(m, "o"):
            m._gko_orig_forward = m.forward
            try:
                is_dec = bool(object.__getattribute__(m, "is_decoder"))
            except AttributeError:
                is_dec = False
            fn = _bind(_t5_forward, tree, _is_decoder=is_dec)
        elif name == "LongformerSelfAttention" and hasattr(m, "query"):
            m._gko_orig_forward = m.forward
            try:
                win = int(object.__getattribute__(m, "one_sided_attn_window_size"))
            except AttributeError:
                win = 0
            fn = _bind(_longformer_forward, tree, _window=win)
        elif name == "MultiHeadSelfAttention" and hasattr(m, "q_lin"):
            fn = _bind(_mha_forward, tree)
        elif name == "Attention" and "BERT_pytorch" in mod:
            fn = _bind(_bert_pytorch_attn, tree)
        elif hasattr(m, "query") and hasattr(m, "key") and hasattr(m, "value") and (
            name.endswith("SelfAttention") or name.endswith("SdpaAttention")
        ) and "DeBERTa" not in name and "Deberta" not in name and "Longformer" not in name:
            m._gko_orig_forward = m.forward
            _flag_module(m)
            fn = _bind(
                _bert_self_forward,
                tree,
                _pos_type=object.__getattribute__(m, "_gko_pos_type"),
                _tuple_out=bool(object.__getattribute__(m, "_gko_tuple_out")),
                _is_decoder=bool(object.__getattribute__(m, "_gko_is_decoder")),
            )
        elif (
            name.endswith("Attention")
            and not name.endswith("SdpaAttention")
            and "Bart" not in name
            and "Whisper" not in name
            and "CLIP" not in name
            and "GPT2" not in name
            and "T5" not in name
            and "Longformer" not in name
            and hasattr(m, "q_proj")
            and hasattr(m, "k_proj")
            and hasattr(m, "v_proj")
            and hasattr(m, "out_proj")
            and hasattr(m, "_shape")
            and getattr(m, "attention_type", "global") != "local"
        ):
            m._gko_orig_forward = m.forward
            _flag_module(m)
            fn = _bind(
                _opt_like_forward,
                tree,
                _causal=bool(object.__getattribute__(m, "_gko_causal")),
                _scale_q=bool(object.__getattribute__(m, "_gko_scale_q")),
            )
        if fn is None:
            continue
        m.forward = types.MethodType(fn, m)
        m._gko_fused_sdpa = True
        n += 1
        kinds.append(name)
    tag = "in_tree_sdpa" if tree else "triton_flash"
    uniq = ",".join(sorted(set(kinds))) if kinds else "none"
    return f"{tag} x{n} ({uniq})"
