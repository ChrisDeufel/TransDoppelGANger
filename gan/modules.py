import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
import math
from torch.nn.init import xavier_uniform_, constant_, xavier_normal_


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False,
                 attn_dim=None, batch_first=True, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = attn_dim if attn_dim is not None else embed_dim
        self.vdim = attn_dim if attn_dim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        #if self._qkv_same_embed_dim is False:
        self.q_proj_weight = Parameter(torch.empty((embed_dim, attn_dim), **factory_kwargs))
        self.k_proj_weight = Parameter(torch.empty((embed_dim, self.kdim), **factory_kwargs))
        self.v_proj_weight = Parameter(torch.empty((embed_dim, self.vdim), **factory_kwargs))
        self.register_parameter('in_proj_weight', None)
        # else:
        #     self.in_proj_weight = Parameter(torch.empty((3 * embed_dim, embed_dim), **factory_kwargs))
        #     self.register_parameter('q_proj_weight', None)
        #     self.register_parameter('k_proj_weight', None)
        #     self.register_parameter('v_proj_weight', None)

        if bias:
            self.in_proj_bias = Parameter(torch.empty(3 * embed_dim, **factory_kwargs))
        else:
            self.register_parameter('in_proj_bias', None)
        self.out_proj = nn.Linear(attn_dim, attn_dim, bias=bias, **factory_kwargs)

        if add_bias_kv:
            self.bias_k = Parameter(torch.empty((1, 1, embed_dim), **factory_kwargs))
            self.bias_v = Parameter(torch.empty((1, 1, embed_dim), **factory_kwargs))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.q_proj_weight)
        xavier_uniform_(self.k_proj_weight)
        xavier_uniform_(self.v_proj_weight)

        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.)
            constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            xavier_normal_(self.bias_v)


    def forward(self, input, key_padding_mask=None,
                need_weights=False, attn_mask=None):
        if self.batch_first:
            input = input.transpose(1, 0)
        q = F.linear(input, self.q_proj_weight.T)
        k = F.linear(input, self.k_proj_weight.T)
        v = F.linear(input, self.v_proj_weight.T)
        tgt_len, bsz, embed_dim = input.shape
        src_len = tgt_len
        # prep attention mask
        """
        make floating point or boolean
        """
        attn_mask = attn_mask.unsqueeze(0)
        #
        # reshape q, k, v for multihead attention and make em batch first
        #
        # q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        # k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        # v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        q = q.contiguous().view(-1, bsz * self.num_heads, self.kdim).transpose(0, 1)
        k = k.contiguous().view(-1, bsz * self.num_heads, self.kdim).transpose(0, 1)
        v = v.contiguous().view(-1, bsz * self.num_heads, self.kdim).transpose(0, 1)

        # update source sequence length after adjustments
        src_len = k.size(1)

        # convert mask to float
        if attn_mask is not None and attn_mask.dtype == torch.bool:
            new_attn_mask = torch.zeros_like(attn_mask, dtype=torch.float)
            new_attn_mask.masked_fill_(attn_mask, float("-inf"))
            attn_mask = new_attn_mask
        #
        # (deep breath) calculate attention and out projection
        #
        attn_output, attn_output_weights = self._scaled_dot_product_attention(q, k, v, attn_mask, self.dropout)
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, self.kdim)
        attn_output = self.out_proj(attn_output)

        if self.batch_first:
            return attn_output.transpose(1, 0), attn_output_weights
        else:
            return attn_output, attn_output_weights

    @staticmethod
    def _scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0):
        B, Nt, E = q.shape
        q = q / math.sqrt(E)
        # (B, Nt, E) x (B, E, Ns) -> (B, Nt, Ns)
        attn = torch.bmm(q, k.transpose(-2, -1))
        if attn_mask is not None:
            attn += attn_mask
        attn = F.softmax(attn, dim=-1)
        if dropout_p > 0.0:
            attn = F.dropout(attn, p=dropout_p)
        # (B, Nt, Ns) x (B, Ns, E) -> (B, Nt, E)
        output = torch.bmm(attn, v)
        return output, attn
