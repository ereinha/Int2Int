# Copyright (c) 2020-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from logging import getLogger
import math
import itertools
import numpy as np
import torch
# from torch._C import _set_backcompat_keepdim_warn
import torch.nn as nn
import torch.nn.functional as F
from .sinekan import SineKANLayer


N_MAX_POSITIONS = 4096  # maximum input sequence length


logger = getLogger()


def Embedding(num_embeddings, embedding_dim, padding_idx=None):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    if padding_idx is not None:
        nn.init.constant_(m.weight[padding_idx], 0)
    return m


def create_sinusoidal_embeddings(n_pos, dim, out):
    position_enc = np.array(
        [
            [pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)]
            for pos in range(n_pos)
        ]
    )
    out[:, 0::2] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
    out[:, 1::2] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
    out.detach_()
    out.requires_grad = False


def gelu(x):
    """
    GELU activation
    https://arxiv.org/abs/1606.08415
    """
    # return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    return 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0)))


def get_masks(slen, lengths, causal):
    """
    Generate hidden states mask, and optionally an attention mask.
    """
    assert lengths.max().item() <= slen
    bs = lengths.size(0)
    alen = torch.arange(slen, dtype=torch.long, device=lengths.device)
    mask = alen < lengths[:, None]

    # attention mask is the same as mask, or triangular inferior attention (causal)
    if causal:
        attn_mask = alen[None, None, :].repeat(bs, slen, 1) <= alen[None, :, None]
    else:
        attn_mask = mask

    # sanity check
    assert mask.size() == (bs, slen)
    assert causal is False or attn_mask.size() == (bs, slen, slen)

    return mask, attn_mask


class MultiHeadAttention(nn.Module):

    NEW_ID = itertools.count()

    def __init__(self, n_heads, dim, src_dim, dropout, normalized_attention, xav_init=False):
        super().__init__()
        self.layer_id = next(MultiHeadAttention.NEW_ID)
        self.dim = dim
        self.src_dim = src_dim
        self.n_heads = n_heads
        self.dropout = dropout
        self.normalized_attention = normalized_attention
        assert self.dim % self.n_heads == 0

        self.q_lin = nn.Linear(dim, dim)
        self.k_lin = nn.Linear(src_dim, dim)
        self.v_lin = nn.Linear(src_dim, dim)
        self.out_lin = nn.Linear(dim, dim)
        if self.normalized_attention:
            self.attention_scale = nn.Parameter(
                torch.tensor(1.0 / math.sqrt(dim // n_heads))
            )
        if xav_init:
            gain = (1 / math.sqrt(2)) if self.src_dim == self.dim else 1.0
            nn.init.xavier_uniform_(self.q_lin.weight, gain=gain)
            nn.init.xavier_uniform_(self.k_lin.weight, gain=gain)
            nn.init.xavier_uniform_(self.v_lin.weight, gain=gain)
            nn.init.xavier_uniform_(self.out_lin.weight)
            nn.init.constant_(self.out_lin.bias, 0.0)

    def forward(self, input, mask, kv=None, use_cache=False, first_loop=True):
        """
        Self-attention (if kv is None)
        or attention over source sentence (provided by kv).
        Input is (bs, qlen, dim)
        Mask is (bs, klen) (non-causal) or (bs, klen, klen)
        """
        assert not (use_cache and self.cache is None)
        bs, qlen, dim = input.size()
        if kv is None:
            klen = qlen if not use_cache else self.cache["slen"] + qlen
        else:
            klen = kv.size(1)
        assert dim == self.dim, "Dimensions do not match: %s input vs %s configured" % (
            dim,
            self.dim,
        )
        n_heads = self.n_heads
        dim_per_head = dim // n_heads
        mask_reshape = (bs, 1, qlen, klen) if mask.dim() == 3 else (bs, 1, 1, klen)

        def shape(x):
            """  projection """
            return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)

        def unshape(x):
            """  compute context """
            return (
                x.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * dim_per_head)
            )

        q = shape(self.q_lin(input))  # (bs, n_heads, qlen, dim_per_head)
        if kv is None:
            k = shape(self.k_lin(input))  # (bs, n_heads, qlen, dim_per_head)
            v = shape(self.v_lin(input))  # (bs, n_heads, qlen, dim_per_head)
        elif not use_cache or self.layer_id not in self.cache:
            k = v = kv
            k = shape(self.k_lin(k))  # (bs, n_heads, qlen, dim_per_head)
            v = shape(self.v_lin(v))  # (bs, n_heads, qlen, dim_per_head)

        if use_cache:
            if self.layer_id in self.cache:
                if kv is None and first_loop:
                    k_, v_ = self.cache[self.layer_id]
                    k = torch.cat([k_, k], dim=2)  # (bs, n_heads, klen, dim_per_head)
                    v = torch.cat([v_, v], dim=2)  # (bs, n_heads, klen, dim_per_head)
                else:
                    k, v = self.cache[self.layer_id]
            self.cache[self.layer_id] = (k, v)
        if self.normalized_attention:
            q = F.normalize(q, p=2, dim=-1)
            k = F.normalize(k, p=2, dim=-1)
            q = q * self.attention_scale
        else:
            q = q / math.sqrt(dim_per_head)  # (bs, n_heads, qlen, dim_per_head)
        scores = torch.matmul(q, k.transpose(2, 3))  # (bs, n_heads, qlen, klen)
        mask = (
            (mask == 0).view(mask_reshape).expand_as(scores)
        )  # (bs, n_heads, qlen, klen)
        scores.masked_fill_(mask, -float("inf"))  # (bs, n_heads, qlen, klen)

        weights = F.softmax(scores.float(), dim=-1).type_as(
            scores
        )  # (bs, n_heads, qlen, klen)
        weights = F.dropout(
            weights, p=self.dropout, training=self.training
        )  # (bs, n_heads, qlen, klen)
        context = torch.matmul(weights, v)  # (bs, n_heads, qlen, dim_per_head)
        context = unshape(context)  # (bs, qlen, dim)

        if TransformerModel.STORE_OUTPUTS and not self.training:
            self.outputs = weights.detach().cpu()

        return self.out_lin(context)


class TransformerFFN(nn.Module):
    def __init__(self, in_dim, dim_hidden, out_dim, hidden_layers, dropout, gelu_activation=False, xav_init=False):
        super().__init__()
        self.dropout = dropout
        self.hidden_layers = hidden_layers
        self.act = gelu if gelu_activation else F.relu
        self.midlin = nn.ModuleList()
        self.lin1 = nn.Linear(in_dim, dim_hidden)
        for i in range(1, self.hidden_layers):
            self.midlin.append(nn.Linear(dim_hidden, dim_hidden))
        self.lin2 = nn.Linear(dim_hidden, out_dim)
        if xav_init:
            nn.init.xavier_uniform_(self.lin1.weight)
            nn.init.constant_(self.lin1.bias, 0.0)
            for mlin in self.midlin:
                nn.init.xavier_uniform_(mlin.weight)
                nn.init.constant_(mlin.bias, 0.0)
            nn.init.xavier_uniform_(self.lin2.weight)
            nn.init.constant_(self.lin2.bias, 0.0)

    def forward(self, input):
        x = self.lin1(input)
        x = self.act(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        for mlin in self.midlin:
            x = mlin(x)
            x = self.act(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        return x


class Gate(nn.Module):
    def __init__(self, dimension, scalar, dropout, biased_gates, gate_bias):
        super().__init__()
        self.dropout = dropout
        self.gate1 = nn.Linear(dimension, 4 * dimension)
        self.gate2 = nn.Linear(4 * dimension, 1 if scalar else dimension)
        if biased_gates:
            self.gate2.bias.data.fill_(gate_bias)

    def forward(self, x):
        outp = self.gate1(x)
        outp = F.relu(outp)
        outp = F.dropout(outp, p=self.dropout, training=self.training)
        outp = self.gate2(outp)
        return torch.sigmoid(outp)


class TransformerLayer(nn.Module):
    def __init__(self, params, is_encoder, gated=False, is_last=False):
        """
        Transformer model (encoder or decoder).
        """
        super().__init__()

        self.is_encoder = is_encoder
        self.is_decoder = not is_encoder
        self.is_last = is_last

        # model parameters
        self.dim = params.enc_emb_dim if is_encoder else params.dec_emb_dim  # 512 by default
        self.src_dim = params.enc_emb_dim
        self.hidden_dim = self.dim * 4  # 2048 by default
        self.n_hidden_layers = params.n_enc_hidden_layers if is_encoder else params.n_dec_hidden_layers
        self.n_heads = params.n_enc_heads if is_encoder else params.n_dec_heads  # 8 by default
        self.n_layers = params.n_enc_layers if is_encoder else params.n_dec_layers
        self.dropout = params.dropout
        self.attention_dropout = params.attention_dropout
        self.gated = gated
        self.scalar_gate = params.scalar_gate
        
        assert (
            self.dim % self.n_heads == 0
        ), "transformer dim must be a multiple of n_heads"
        self.self_attention = MultiHeadAttention(
            self.n_heads,
            self.dim,
            self.dim,
            dropout=self.attention_dropout,
            normalized_attention=params.norm_attention,
        )
        self.layer_norm1 = nn.LayerNorm(self.dim, eps=1e-12)
        if self.is_decoder:
            self.layer_norm15 = nn.LayerNorm(self.dim, eps=1e-12)
            self.cross_attention = MultiHeadAttention(
                self.n_heads,
                self.dim,
                self.src_dim,
                dropout=self.attention_dropout,
                normalized_attention=params.norm_attention,
            )
        if self.is_decoder and self.is_last:
            self.ffn = SineKANLayer(
                self.dim,
                self.dim,
            )
        else:
            self.ffn = TransformerFFN(
                self.dim,
                self.hidden_dim,
                self.dim,
                self.n_hidden_layers,
                dropout=self.dropout,
                gelu_activation = params.gelu_activation
            )
            
        self.layer_norm2 = nn.LayerNorm(self.dim, eps=1e-12)
        if self.gated:
            self.gate = Gate(self.dim, self.scalar_gate, self.dropout, params.biased_gates, params.gate_bias)

    def forward(self, x, attn_mask, src_mask, src_enc, use_cache=False, cache=None, loop_count=1):
        tensor = x
        for i in range(loop_count):
            # self attention
            self.self_attention.cache = cache
            attn = self.self_attention(tensor, attn_mask, use_cache=use_cache, first_loop=(i==0))
            attn = F.dropout(attn, p=self.dropout, training=self.training)
            output = tensor + attn
            output = self.layer_norm1(output)

            if self.gated:
                gate = self.gate(output)
            
            # encoder attention (for decoder only)
            if self.is_decoder and src_enc is not None:
                self.cross_attention.cache = cache
                attn = self.cross_attention(
                    tensor, src_mask, kv=src_enc, use_cache=use_cache, first_loop=(i==0)
                )
                attn = F.dropout(attn, p=self.dropout, training=self.training)
                output = output + attn
                output = self.layer_norm15(output)

            # FFN
            if self.is_last and self.is_decoder:
                output = self.ffn(output)
            else:
                output = output + self.ffn(output)
            output = self.layer_norm2(output)
            if self.gated:
                tensor = gate * output + (1 - gate) * tensor
            else:
                tensor = output
        return tensor


class AdaptiveHalt(nn.Module):
    def __init__(self, params, is_encoder, gated):
        super().__init__()
        self.dim = params.enc_emb_dim if is_encoder else params.dec_emb_dim  # 512 by default
        self.max_loops = params.enc_loops if is_encoder else params.dec_loops
        assert params.act_threshold >= 0
        self.threshold = 1.0 - params.act_threshold
        self.halt_prob = nn.Linear(self.dim, 1)
        if params.act_biased:
            self.halt_prob.bias.data.fill_(params.act_bias)
        ponder = params.act_ponder_coupling
        self.ponder_coupling = ponder
        self.ponder_penalty = 0
        self.layer = TransformerLayer(params, is_encoder, gated)

    def forward(self, input, attn_mask, src_mask, src_enc, use_cache, cache, loop_count):
        bs = input.size(0)
        slen = input.size(1)
        shape = (bs, slen)
        halting_probability = torch.zeros(shape, device=input.device)
        remainders = torch.zeros_like(halting_probability)
        # n_updates = torch.zeros_like(halting_probability)
        acc_state = 0
        
        for i in range(self.max_loops):
            # stop probability for current state
            p = torch.squeeze(torch.sigmoid(self.halt_prob(input)), -1)
            # running tokens at step start
            still_running = torch.less(halting_probability, 1.0).float()
            # stopping this step
            new_halted = torch.greater(halting_probability + p * still_running, self.threshold).float() * still_running
            # running at step end
            still_running = torch.less_equal(halting_probability + p * still_running, self.threshold).float() * still_running

            halting_probability += p * still_running
            # R(t) in ACT paper
            remainders += new_halted * (1 - halting_probability)
            halting_probability += new_halted * remainders
            # N(t) in ACT paper (unused)
            # n_updates += still_running + new_halted
            # update state
            input = self.layer.forward(input, attn_mask, src_mask, src_enc, use_cache, cache, loop_count)
            # weighted final state
            update_weights = torch.unsqueeze(p * still_running + new_halted * remainders, -1)
            acc_state = (input * update_weights) + (acc_state * (1 - update_weights))
            if still_running.sum() == 0:
                break
        
        remainders += torch.less(halting_probability, 1.0).float() * (1 - halting_probability)
        self.ponder_penalty = self.ponder_coupling * torch.mean(remainders)
        return acc_state



class TransformerModel(nn.Module):

    STORE_OUTPUTS = False

    def __init__(self, params, id2word, is_encoder, with_output, sinekan=False):
        """
        Transformer model (encoder or decoder).
        """
        super().__init__()

        # encoder / decoder, output layer
        self.dtype = torch.half if params.fp16 else torch.float
        self.is_encoder = is_encoder
        self.is_decoder = not is_encoder
        self.sinekan = sinekan
        self.with_output = with_output
        self.decoder_only = params.architecture == "decoder_only"

        self.xav_init = params.xav_init

        # dictionary
        self.n_words = params.n_words
        self.eos_index = params.eos_index
        self.pad_index = params.pad_index
        self.sep_index = params.sep_index

        self.id2word = id2word
        assert len(self.id2word) == self.n_words

        # model parameters
        self.dim = params.enc_emb_dim if is_encoder else params.dec_emb_dim  # 512 by default
        self.src_dim = params.enc_emb_dim
        self.max_src_len = params.max_src_len
        self.hidden_dim = self.dim * 4  # 2048 by default
        self.n_hidden_layers = params.n_enc_hidden_layers if is_encoder else params.n_dec_hidden_layers
        self.n_heads = params.n_enc_heads if is_encoder else params.n_dec_heads  # 8 by default
        self.n_layers = params.n_enc_layers if is_encoder else params.n_dec_layers
        self.has_pos_emb = params.enc_has_pos_emb if is_encoder else params.dec_has_pos_emb
        self.dropout = params.dropout
        self.attention_dropout = params.attention_dropout
        self.norm_attention = params.norm_attention
        assert (
            self.dim % self.n_heads == 0
        ), "transformer dim must be a multiple of n_heads"

        # iteration 
        self.loop_idx = params.enc_loop_idx if is_encoder else params.dec_loop_idx
        assert self.loop_idx < self.n_layers, "loop idx must be lower than nr of layers" 
        self.loops = params.enc_loops if is_encoder else params.dec_loops
        
        self.act = params.enc_act if is_encoder else params.dec_act
        assert (not self.act) or (self.loop_idx >= 0)
        
        # embeddings
        if self.has_pos_emb:
            self.position_embeddings = Embedding(N_MAX_POSITIONS, self.dim)
            if params.sinusoidal_embeddings:
                create_sinusoidal_embeddings(
                    N_MAX_POSITIONS, self.dim, out=self.position_embeddings.weight
                )
        self.embeddings = Embedding(self.n_words, self.dim, padding_idx=self.pad_index)
        self.layer_norm_emb = nn.LayerNorm(self.dim, eps=1e-12)

        # transformer layers
        self.layers = nn.ModuleList()
        for layer_id in range(self.n_layers):
            if params.enc_gated and self.is_encoder:
                gated = True
            elif params.dec_gated and self.is_decoder:
                gated = True
            elif params.gated and layer_id == self.loop_idx:
                gated = True
            else:
                gated = False
            if self.act and layer_id == self.loop_idx:
                self.layers.append(AdaptiveHalt(params, self.is_encoder, gated))
            else:
                if self.sinekan:
                    self.layers.append(TransformerLayer(params, self.is_encoder, gated, is_last=layer_id==self.n_layers-1))
                else:
                    self.layers.append(TransformerLayer(params, self.is_encoder, gated))
        self.cache = None

        # output layer
        if self.with_output:
            self.proj = nn.Linear(self.dim, params.n_words, bias=True)
            if self.xav_init:
                nn.init.xavier_uniform_(self.proj.weight)
                nn.init.constant_(self.proj.bias, 0.0)
            if params.share_inout_emb:
                self.proj.weight = self.embeddings.weight

    def forward(self, mode, **kwargs):
        """
        Forward function with different forward modes.
        ### Small hack to handle PyTorch distributed.
        """
        if mode == "fwd":
            return self.fwd(**kwargs)
        elif mode == "predict":
            return self.predict(**kwargs)
        else:
            raise Exception("Unknown mode: %s" % mode)

    def fwd(
        self,
        x,
        lengths,
        causal,
        src_enc=None,
        src_len=None,
        positions=None,
        use_cache=False,
    ):
        """
        Inputs:
            `x` LongTensor(slen, bs), containing word indices
            `lengths` LongTensor(bs), containing the length of each sentence
            `causal` Boolean, if True, the attention is only done over previous hidden states
            `positions` LongTensor(slen, bs), containing word positions
        """
        # lengths = (x != self.pad_index).float().sum(dim=1)
        # mask = x != self.pad_index

        # check inputs
        slen, bs = x.size()
        assert lengths.size(0) == bs
        assert lengths.max().item() <= slen
        x = x.transpose(0, 1)  # batch size as dimension 0
        assert (src_enc is None) == (src_len is None)
        if src_enc is not None:
            assert self.is_decoder
            assert src_enc.size(0) == bs
        assert not (use_cache and self.cache is None)

        # generate masks
        mask, attn_mask = get_masks(slen, lengths, causal)
        src_mask = None
        if self.is_decoder and (src_enc is not None):
            if self.max_src_len > 0:
                src_mask = (
                    torch.arange(src_len.max(), dtype=torch.long, device=lengths.device)
                    < torch.clamp(src_len[:, None], max=self.max_src_len)
                )
            else:
                src_mask = (
                    torch.arange(src_len.max(), dtype=torch.long, device=lengths.device)
                    < src_len[:, None]
                )
        
        # positions
        if positions is None:
            positions = x.new(slen).long()
            positions = torch.arange(slen, out=positions).unsqueeze(0)
        else:
            assert positions.size() == (slen, bs)
            positions = positions.transpose(0, 1)

        # do not recompute cached elements
        if use_cache:
            _slen = slen - self.cache["slen"]
            x = x[:, -_slen:]
            positions = positions[:, -_slen:]
            mask = mask[:, -_slen:]
            attn_mask = attn_mask[:, -_slen:]

        # all layer outputs
        if TransformerModel.STORE_OUTPUTS and not self.training:
            self.outputs = []

        # embeddings
        tensor = self.embeddings(x)
        if self.has_pos_emb:
            tensor = tensor + self.position_embeddings(positions).expand_as(tensor)
        tensor = self.layer_norm_emb(tensor)
        tensor = F.dropout(tensor, p=self.dropout, training=self.training)
        tensor *= mask.unsqueeze(-1).to(tensor.dtype)
        if TransformerModel.STORE_OUTPUTS and not self.training:
            self.outputs.append(tensor.detach().cpu())

        # transformer layers
        for i in range(self.n_layers):
            loops = 1
            if self.loop_idx == -2 or self.loop_idx == i:
                loops = self.loops
            tensor = self.layers[i].forward(tensor, attn_mask, src_mask, src_enc, use_cache=use_cache, cache=self.cache, loop_count=loops)
    
            tensor *= mask.unsqueeze(-1).to(tensor.dtype)
            if TransformerModel.STORE_OUTPUTS and not self.training:
                self.outputs.append(tensor.detach().cpu())

        # update cache length
        if use_cache:
            self.cache["slen"] += tensor.size(1)

        # move back sequence length to dimension 0
        tensor = tensor.transpose(0, 1)

        return tensor

    def predict(self, tensor, pred_mask, y, get_scores):
        """
        Given the last hidden state, compute word scores and/or the loss.
            `pred_mask` is a ByteTensor of shape (slen, bs), filled with 1 when
                we need to predict a word
            `y` is a LongTensor of shape (pred_mask.sum(),)
            `get_scores` is a boolean specifying whether we need to return scores
        """
        x = tensor[pred_mask.unsqueeze(-1).expand_as(tensor)].view(-1, self.dim)
        assert (y == self.pad_index).sum().item() == 0
        scores = self.proj(x).view(-1, self.n_words)
        loss = F.cross_entropy(scores.float(), y, reduction="mean")
        return scores, loss

    def decode(self, src_enc, src_len, exp_len):
         # input batch
        bs = len(src_len)
        assert src_enc.size(1) == bs

        # generated sentences
        max_len = src_enc.size(0)

        tensor = self.fwd(x=src_enc, lengths=src_len, causal=False)
        assert tensor.size() == (max_len, bs, self.dim)
        scores = self.proj(tensor)  # (len, bs, n_words)
        # select next words: sample or greedy
        sample_temperature = None
        if sample_temperature is None:
            next_words = torch.topk(scores, 1)[1].squeeze(2)
        else:
            next_words = torch.multinomial(
                F.softmax(scores.float() / sample_temperature, dim=1), 1
            ).squeeze(2)
        assert next_words.size() == (max_len, bs,)
        next_words = next_words[:exp_len] 
        return next_words.transpose(0,1)


    def generate(self, src_enc, src_len, max_len=200, sample_temperature=None):
        """
        Decode a sentence given initial start.
        `x`:
            - LongTensor(bs, slen)
                <EOS> W1 W2 W3 <EOS> <PAD>
                <EOS> W1 W2 W3   W4  <EOS>
        `lengths`:
            - LongTensor(bs) [5, 6]
        `positions`:
            - False, for regular "arange" positions (LM)
            - True, to reset positions from the new generation (MT)
        """

        # input batch
        bs = len(src_len)
        #assert src_enc.size(0) == bs

        # generated sentences
        generated = src_len.new(max_len, bs)  # upcoming output
        generated.fill_(self.pad_index)  # fill upcoming ouput with <PAD>
        # current position / max lengths / length of generated sentences 
        if self.decoder_only:
            max_src = src_len.max()
            generated[:max_src] = src_enc
            generated[max_src-1] = self.sep_index
        #    cur_len = src_len.min()
        #    gen_len = src_len.clone().fill_(cur_len)
        else:
            generated[0].fill_(self.eos_index)  # we use <EOS> for <BOS> everywhere
        cur_len = 1
        gen_len = src_len.clone().fill_(1)

        # positions
        positions = src_len.new(max_len).long()
        positions = (
            torch.arange(max_len, out=positions).unsqueeze(1).expand_as(generated)
        )

        unfinished_sents = src_len.clone().fill_(1)

        # cache compute states
        self.cache = {"slen": 0}

        while cur_len < max_len:

            if self.decoder_only:
                # compute word scores
                tensor = self.forward(
                    "fwd",
                    x=generated[:cur_len],
                    lengths=gen_len,
                    positions=positions[:cur_len],
                    causal=True,
                    src_enc=None,
                    src_len=None,
                    use_cache=True,
                )
            else:
                # compute word scores
                tensor = self.forward(
                    "fwd",
                    x=generated[:cur_len],
                    lengths=gen_len,
                    positions=positions[:cur_len],
                    causal=True,
                    src_enc=src_enc,
                    src_len=src_len,
                    use_cache=True,
                )
            assert tensor.size() == (1, bs, self.dim)
            tensor = tensor.data[-1, :, :] # .to(self.dtype)  # (bs, dim)
            scores = self.proj(tensor)  # (bs, n_words)

            # select next words: sample or greedy
            if sample_temperature is None:
                next_words = torch.topk(scores, 1)[1].squeeze(1)
            else:
                next_words = torch.multinomial(
                    F.softmax(scores.float() / sample_temperature, dim=1), 1
                ).squeeze(1)
            assert next_words.size() == (bs,)

            # update generations / lengths / finished sentences / current length
            if self.decoder_only:
                up_mask = (src_len <= cur_len)
                to_update = unfinished_sents * up_mask 
                generated[cur_len] = next_words * to_update + (1 - to_update) * (
                    generated[cur_len] * ~up_mask + self.pad_index * up_mask
                )
                gen_len.add_(unfinished_sents)
                unfinished_sents.mul_((~up_mask).long() + up_mask * next_words.ne(self.eos_index).long())
            else:
                generated[cur_len] = next_words * unfinished_sents + self.pad_index * (
                    1 - unfinished_sents
                )
                gen_len.add_(unfinished_sents)
                unfinished_sents.mul_(next_words.ne(self.eos_index).long())

            cur_len = cur_len + 1

            # stop when there is a </s> in each sentence, or if we exceed the maximal length
            if unfinished_sents.max() == 0:
                break

        # add <EOS> to unfinished sentences
        if cur_len == max_len:
            generated[-1].masked_fill_(unfinished_sents.bool(), self.eos_index)

        # sanity check
        assert (generated == self.eos_index).sum() == 2 * bs

        return generated[:cur_len].cpu(), gen_len.cpu()

    def generate_beam(
        self, src_enc, src_len, beam_size, length_penalty, early_stopping, max_len=200
    ):
        """
        Decode a sentence given initial start.
        `x`:
            - LongTensor(bs, slen)
                <EOS> W1 W2 W3 <EOS> <PAD>
                <EOS> W1 W2 W3   W4  <EOS>
        `lengths`:
            - LongTensor(bs) [5, 6]
        `positions`:
            - False, for regular "arange" positions (LM)
            - True, to reset positions from the new generation (MT)
        """

        # check inputs
        assert src_enc.size(0) == src_len.size(0)
        assert beam_size >= 1

        # batch size / number of words
        bs = len(src_len)
        n_words = self.n_words

        # expand to beam size the source latent representations / source lengths
        src_enc = (
            src_enc.unsqueeze(1)
            .expand((bs, beam_size) + src_enc.shape[1:])
            .contiguous()
            .view((bs * beam_size,) + src_enc.shape[1:])
        )
        src_len = src_len.unsqueeze(1).expand(bs, beam_size).contiguous().view(-1)

        # generated sentences (batch with beam current hypotheses)
        generated = src_len.new(max_len, bs * beam_size)  # upcoming output
        generated.fill_(self.pad_index)  # fill upcoming ouput with <PAD>
        generated[0].fill_(self.eos_index)  # we use <EOS> for <BOS> everywhere

        # generated hypotheses
        generated_hyps = [
            BeamHypotheses(beam_size, max_len, length_penalty, early_stopping)
            for _ in range(bs)
        ]

        # positions
        positions = src_len.new(max_len).long()
        positions = (
            torch.arange(max_len, out=positions).unsqueeze(1).expand_as(generated)
        )

        # scores for each sentence in the beam
        beam_scores = src_enc.new(bs, beam_size).float().fill_(0)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view(-1)

        # current position
        cur_len = 1

        # cache compute states
        self.cache = {"slen": 0}

        # done sentences
        done = [False for _ in range(bs)]

        while cur_len < max_len:

            # compute word scores
            tensor = self.forward(
                "fwd",
                x=generated[:cur_len],
                lengths=src_len.new(bs * beam_size).fill_(cur_len),
                positions=positions[:cur_len],
                causal=True,
                src_enc=src_enc,
                src_len=src_len,
                use_cache=True,
            )

            assert tensor.size() == (1, bs * beam_size, self.dim)
            tensor = tensor.data[-1, :, :]  # .to(self.dtype)  # (bs * beam_size, dim)
            scores = self.proj(tensor)  # (bs * beam_size, n_words)
            scores = F.log_softmax(scores.float(), dim=-1)  # (bs * beam_size, n_words)
            assert scores.size() == (bs * beam_size, n_words)

            # select next words with scores
            _scores = scores + beam_scores[:, None].expand_as(
                scores
            )  # (bs * beam_size, n_words)
            _scores = _scores.view(bs, beam_size * n_words)  # (bs, beam_size * n_words)

            next_scores, next_words = torch.topk(
                _scores, 2 * beam_size, dim=1, largest=True, sorted=True
            )
            assert next_scores.size() == next_words.size() == (bs, 2 * beam_size)

            # next batch beam content
            # list of (bs * beam_size) tuple(next hypothesis score, next word, current position in the batch)
            next_batch_beam = []

            # for each sentence
            for sent_id in range(bs):

                # if we are done with this sentence
                done[sent_id] = done[sent_id] or generated_hyps[sent_id].is_done(
                    next_scores[sent_id].max().item()
                )
                if done[sent_id]:
                    next_batch_beam.extend(
                        [(0, self.pad_index, 0)] * beam_size
                    )  # pad the batch
                    continue

                # next sentence beam content
                next_sent_beam = []

                # next words for this sentence
                for idx, value in zip(next_words[sent_id], next_scores[sent_id]):

                    # get beam and word IDs
                    beam_id = idx // n_words
                    word_id = idx % n_words

                    # end of sentence, or next word
                    if word_id == self.eos_index or cur_len + 1 == max_len:
                        generated_hyps[sent_id].add(
                            generated[:cur_len, sent_id * beam_size + beam_id]
                            .clone()
                            .cpu(),
                            value.item(),
                        )
                    else:
                        next_sent_beam.append(
                            (value, word_id, sent_id * beam_size + beam_id)
                        )

                    # the beam for next step is full
                    if len(next_sent_beam) == beam_size:
                        break

                # update next beam content
                assert len(next_sent_beam) == 0 if cur_len + 1 == max_len else beam_size
                if len(next_sent_beam) == 0:
                    next_sent_beam = [
                        (0, self.pad_index, 0)
                    ] * beam_size  # pad the batch
                next_batch_beam.extend(next_sent_beam)
                assert len(next_batch_beam) == beam_size * (sent_id + 1)

            # sanity check / prepare next batch
            assert len(next_batch_beam) == bs * beam_size
            beam_scores = beam_scores.new([x[0] for x in next_batch_beam])
            beam_words = generated.new([x[1] for x in next_batch_beam])
            beam_idx = src_len.new([x[2] for x in next_batch_beam])

            # re-order batch and internal states
            generated = generated[:, beam_idx]
            generated[cur_len] = beam_words
            for k in self.cache.keys():
                if k != "slen":
                    self.cache[k] = (
                        self.cache[k][0][beam_idx],
                        self.cache[k][1][beam_idx],
                    )

            # update current length
            cur_len = cur_len + 1

            # stop when we are done with each sentence
            if all(done):
                break

        # def get_coeffs(s):
        #     roots = [int(s[i + 2]) for i, c in enumerate(s) if c == 'x']
        #     poly = np.poly1d(roots, r=True)
        #     coeffs = list(poly.coefficients.astype(np.int64))
        #     return [c % 10 for c in coeffs], coeffs

        # visualize hypotheses
        # print([len(x) for x in generated_hyps], cur_len)
        # globals().update( locals() );
        # !import code; code.interact(local=vars())
        # for ii in range(bs):
        #     for ss, ww in sorted(generated_hyps[ii].hyp, key=lambda x: x[0], reverse=True):
        #         hh = " ".join(self.id2word[x] for x in ww.tolist())
        #         print(f"{ss:+.4f} {hh}")
        #         # cc = get_coeffs(hh[4:])
        #         # print(f"{ss:+.4f} {hh} || {cc[0]} || {cc[1]}")
        #     print("")

        # select the best hypotheses
        tgt_len = src_len.new(bs)
        best = []

        for i, hypotheses in enumerate(generated_hyps):
            best_hyp = max(hypotheses.hyp, key=lambda x: x[0])[1]
            tgt_len[i] = len(best_hyp) + 1  # +1 for the <EOS> symbol
            best.append(best_hyp)

        # generate target batch
        decoded = src_len.new(tgt_len.max().item(), bs).fill_(self.pad_index)
        for i, hypo in enumerate(best):
            decoded[: tgt_len[i] - 1, i] = hypo
            decoded[tgt_len[i] - 1, i] = self.eos_index

        # sanity check
        assert (decoded == self.eos_index).sum() == 2 * bs

        return decoded, tgt_len, generated_hyps


class BeamHypotheses(object):
    def __init__(self, n_hyp, max_len, length_penalty, early_stopping):
        """
        Initialize n-best list of hypotheses.
        """
        self.max_len = max_len - 1  # ignoring <BOS>
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping
        self.n_hyp = n_hyp
        self.hyp = []
        self.worst_score = 1e9

    def __len__(self):
        """
        Number of hypotheses in the list.
        """
        return len(self.hyp)

    def add(self, hyp, sum_logprobs):
        """
        Add a new hypothesis to the list.
        """
        score = sum_logprobs / len(hyp) ** self.length_penalty
        if len(self) < self.n_hyp or score > self.worst_score:
            self.hyp.append((score, hyp))
            if len(self) > self.n_hyp:
                sorted_scores = sorted(
                    [(s, idx) for idx, (s, _) in enumerate(self.hyp)]
                )
                del self.hyp[sorted_scores[0][1]]
                self.worst_score = sorted_scores[1][0]
            else:
                self.worst_score = min(score, self.worst_score)

    def is_done(self, best_sum_logprobs):
        """
        If there are enough hypotheses and that none of the hypotheses being generated
        can become better than the worst one in the heap,
        then we are done with this sentence.
        """
        if len(self) < self.n_hyp:
            return False
        elif self.early_stopping:
            return True
        else:
            return (
                self.worst_score
                >= best_sum_logprobs / self.max_len ** self.length_penalty
            )
