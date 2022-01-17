import torch
import torch.nn as nn
import torch.nn.functional as F
from output import OutputType, Normalization
from gan.gan_util import init_weights
import math
from torch.nn.parameter import Parameter
from gan.modules import MultiHeadAttention


# torch.utils.backcompat.broadcast_warning.enabled = True
# Transformer Discriminator

class TransformerDiscriminator(nn.Module):
    def __init__(self, input_feature_shape, input_attribute_shape, num_units=200, num_heads=1, scope_name="transformer_discriminator", *args, **kwargs):
        super(TransformerDiscriminator, self).__init__()
        self.scope_name = scope_name
        # only saved for adding to summary writer (see trainer.train)
        self.input_feature_shape = input_feature_shape
        self.input_attribute_shape = input_attribute_shape
        self.num_features = input_feature_shape[2]
        self.positional_encoding = PositionalEncoding(d_model=self.num_features)
        #self.decoder = torch.nn.TransformerDecoderLayer(d_model=attn_dim, nhead=num_heads, dim_feedforward=attn_dim,
        #                                                batch_first=True)
        self.encoder = torch.nn.TransformerEncoderLayer(d_model=self.num_features, nhead=num_heads,
                                                        dim_feedforward=num_units, batch_first=True)
        self.input_ff = input_feature_shape[1] * input_feature_shape[2] + input_attribute_shape[1]
        modules = [nn.Linear(self.input_ff, num_units), nn.ReLU(),
                   nn.Linear(num_units, 1)]
        # https://discuss.pytorch.org/t/append-for-nn-sequential-or-directly-converting-nn-modulelist-to-nn-sequential/7104
        self.disc = nn.Sequential(*modules)
        # initialize weights
        self.disc.apply(init_weights)

    def forward(self, input_feature, input_attribute):
        input_feature = input_feature.transpose(1, 0)
        input_feature = self.positional_encoding(input_feature)
        input_feature = input_feature.transpose(1, 0)
        # TODO: generate Attention Mask
        transformer_output = self.encoder(input_feature)
        transformer_output = torch.flatten(transformer_output, start_dim=1, end_dim=2)
        #input_attribute = torch.flatten(input_attribute, start_dim=1, end_dim=1)
        x = torch.cat((input_attribute, transformer_output), dim=1)
        return self.disc(x)


# Discriminator
class Discriminator(nn.Module):
    def __init__(self, input_feature_shape, input_attribute_shape, num_layers=5, num_units=200,
                 scope_name="discriminator", *args, **kwargs):
        super(Discriminator, self).__init__()
        self.scope_name = scope_name
        # only saved for adding to summary writer (see trainer.train)
        self.input_feature_shape = input_feature_shape
        self.input_attribute_shape = input_attribute_shape
        self.input_size = input_feature_shape[1] * input_feature_shape[2] + input_attribute_shape[1]
        modules = [nn.Linear(self.input_size, num_units), nn.ReLU()]
        for i in range(num_layers - 2):
            modules.append(nn.Linear(num_units, num_units))
            modules.append(nn.ReLU())
        modules.append(nn.Linear(num_units, 1))
        # https://discuss.pytorch.org/t/append-for-nn-sequential-or-directly-converting-nn-modulelist-to-nn-sequential/7104
        self.disc = nn.Sequential(*modules)
        # initialize weights
        self.disc.apply(init_weights)

    def forward(self, input_feature, input_attribute):
        input_feature = torch.flatten(input_feature, start_dim=1, end_dim=2)
        input_attribute = torch.flatten(input_attribute, start_dim=1, end_dim=1)
        x = torch.cat((input_feature, input_attribute), dim=1)
        return self.disc(x)


class AttrDiscriminator(nn.Module):
    def __init__(self, input_attribute_shape, num_layers=5, num_units=200, scope_name="attrDiscriminator", *args,
                 **kwargs):
        super(AttrDiscriminator, self).__init__()
        self.scope_name = scope_name
        # only saved for adding to summary writer (see trainer.train)
        self.input_size = input_attribute_shape[1]
        modules = [nn.Linear(self.input_size, num_units), nn.ReLU()]
        for i in range(num_layers - 2):
            modules.append(nn.Linear(num_units, num_units))
            modules.append(nn.ReLU())
        modules.append(nn.Linear(num_units, 1))

        self.attrdisc = nn.Sequential(*modules)
        # initialize weights
        self.attrdisc.apply(init_weights)

    def forward(self, x):
        return self.attrdisc(x)


class DoppelGANgerGenerator(nn.Module):
    def __init__(self, noise_dim, attribute_outputs, real_attribute_mask, device,
                 attribute_num_units=100, attribute_num_layers=3, scope_name="DoppelGANgerGenerator", *args, **kwargs):
        super(DoppelGANgerGenerator, self).__init__()
        self.scope_name = scope_name
        self.device = device
        # calculate attribute dimensions
        self.real_attribute_dim = 0
        self.addi_attribute_dim = 0
        for i in range(len(attribute_outputs)):
            if real_attribute_mask[i]:
                self.real_attribute_dim += attribute_outputs[i].dim
            else:
                self.addi_attribute_dim += attribute_outputs[i].dim

        # build real attribute generator
        modules = [nn.Linear(noise_dim, attribute_num_units), nn.ReLU(),
                   nn.BatchNorm1d(num_features=attribute_num_units, eps=1e-5, momentum=0.9)]
        for i in range(attribute_num_layers - 2):
            modules.append(nn.Linear(attribute_num_units, attribute_num_units))
            modules.append(nn.ReLU())
            modules.append(nn.BatchNorm1d(num_features=attribute_num_units, eps=1e-5, momentum=0.9))
        self.real_attribute_gen = nn.Sequential(*modules)
        # initialize weights
        self.real_attribute_gen.apply(init_weights)

        # build additive attribute generator
        modules = [nn.Linear(noise_dim + self.real_attribute_dim, attribute_num_units), nn.ReLU(),
                   nn.BatchNorm1d(num_features=attribute_num_units, eps=1e-5, momentum=0.9)]
        for i in range(attribute_num_layers - 2):
            modules.append(nn.Linear(attribute_num_units, attribute_num_units))
            modules.append(nn.ReLU())
            modules.append(nn.BatchNorm1d(num_features=attribute_num_units, eps=1e-5, momentum=0.9))
        self.addi_attribute_gen = nn.Sequential(*modules)
        # initialize weights
        self.addi_attribute_gen.apply(init_weights)

        # create real and additive generator output layers
        self.real_attr_output_layers = nn.ModuleList()
        self.addi_attr_output_layers = nn.ModuleList()
        for i in range(len(attribute_outputs)):
            modules = [nn.Linear(attribute_num_units, attribute_outputs[i].dim)]
            if attribute_outputs[i].type_ == OutputType.DISCRETE:
                modules.append(nn.Softmax(dim=-1))
            else:
                if attribute_outputs[i].normalization == Normalization.ZERO_ONE:
                    modules.append(nn.Sigmoid())
                else:
                    modules.append(nn.Tanh())
            if real_attribute_mask[i]:
                self.real_attr_output_layers.append(nn.Sequential(*modules))
            else:
                self.addi_attr_output_layers.append(nn.Sequential(*modules))
        # initialize weights
        self.real_attr_output_layers.apply(init_weights)
        self.addi_attr_output_layers.apply(init_weights)

    def forward(self, real_attribute_noise, addi_attribute_noise, feature_input_noise):
        all_attribute = []
        all_discrete_attribute = []
        # real attribute generator
        real_attribute_gen_output = self.real_attribute_gen(real_attribute_noise)
        part_attribute = []
        part_discrete_attribute = []
        for attr_layer in self.real_attr_output_layers:
            sub_output = attr_layer(real_attribute_gen_output)
            if isinstance(attr_layer[-1], nn.Softmax):
                sub_output_discrete = F.one_hot(torch.argmax(sub_output, dim=1), num_classes=sub_output.shape[1])
            else:
                sub_output_discrete = sub_output
            part_attribute.append(sub_output)
            part_discrete_attribute.append(sub_output_discrete)
        part_attribute = torch.cat(part_attribute, dim=1)
        part_discrete_attribute = torch.cat(part_discrete_attribute, dim=1)
        part_discrete_attribute = part_discrete_attribute.detach()
        all_attribute.append(part_attribute)
        all_discrete_attribute.append(part_discrete_attribute)

        # create addi attribute generator input
        addi_attribute_input = torch.cat((part_discrete_attribute, addi_attribute_noise), dim=1)
        # addi_attribute_input = torch.cat((part_attribute, addi_attribute_noise), dim=1)
        # add attribute generator
        addi_attribute_gen_output = self.addi_attribute_gen(addi_attribute_input)
        part_attribute = []
        part_discrete_attribute = []
        for addi_attr_layer in self.addi_attr_output_layers:
            sub_output = addi_attr_layer(addi_attribute_gen_output)
            if isinstance(addi_attr_layer[-1], nn.Softmax):
                sub_output_discrete = F.one_hot(torch.argmax(sub_output, dim=1), num_classes=sub_output.shape[1])
            else:
                sub_output_discrete = sub_output
            part_attribute.append(sub_output)
            part_discrete_attribute.append(sub_output_discrete)
        part_attribute = torch.cat(part_attribute, dim=1)
        part_discrete_attribute = torch.cat(part_discrete_attribute, dim=1)
        part_discrete_attribute = part_discrete_attribute.detach()
        all_attribute.append(part_attribute)
        all_discrete_attribute.append(part_discrete_attribute)
        all_attribute = torch.cat(all_attribute, dim=1)
        all_discrete_attribute = torch.cat(all_discrete_attribute, dim=1)

        # create feature generator input
        attribute_output = torch.unsqueeze(all_discrete_attribute, dim=1)
        # attribute_output = all_discrete_attribute
        attribute_feature_input = torch.cat(feature_input_noise.shape[1] * [attribute_output], dim=1)
        attribute_feature_input = attribute_feature_input.detach()
        feature_gen_input = torch.cat((attribute_feature_input, feature_input_noise), dim=2)
        return all_attribute, feature_gen_input


class DoppelGANgerGeneratorRNN(DoppelGANgerGenerator):
    def __init__(self, noise_dim, feature_outputs, attribute_outputs, real_attribute_mask, device, sample_len,
                 attribute_num_units=100, attribute_num_layers=3, feature_num_units=100,
                 feature_num_layers=1, scope_name="DoppelGANgerGenerator", *args, **kwargs):
        super().__init__(noise_dim, attribute_outputs, real_attribute_mask, device, attribute_num_units,
                         attribute_num_layers,
                         scope_name)
        self.device = device
        self.feature_dim = 0
        for feature in feature_outputs:
            self.feature_dim += feature.dim
        self.feature_num_layers = feature_num_layers
        self.feature_num_units = feature_num_units
        # create feature generator
        self.feature_rnn = nn.LSTM(input_size=noise_dim + self.real_attribute_dim + self.addi_attribute_dim,
                                   hidden_size=feature_num_units,
                                   num_layers=feature_num_layers,
                                   batch_first=True)
        # initialize weights
        self.feature_rnn.apply(init_weights)

        # create feature output layers
        self.feature_output_layers = nn.ModuleList()
        feature_counter = 0
        feature_len = len(feature_outputs)
        for i in range(len(feature_outputs) * sample_len):
            modules = [nn.Linear(feature_num_units, feature_outputs[feature_counter].dim)]
            if feature_outputs[feature_counter].type_ == OutputType.DISCRETE:
                modules.append(nn.Softmax(dim=-1))
            else:
                if feature_outputs[feature_counter].normalization == Normalization.ZERO_ONE:
                    modules.append(nn.Sigmoid())
                else:
                    modules.append(nn.Tanh())
            feature_counter += 1
            if feature_counter % feature_len == 0:
                feature_counter = 0
            self.feature_output_layers.append(nn.Sequential(*modules))
        # initialize weights
        self.feature_output_layers.apply(init_weights)

    def forward(self, real_attribute_noise, addi_attribute_noise, feature_input_noise):
        all_attribute, feature_gen_input = super().forward(real_attribute_noise, addi_attribute_noise,
                                                           feature_input_noise)
        # initial hidden and cell state
        h_o = torch.randn((self.feature_num_layers, feature_gen_input.size(0), self.feature_num_units)).to(self.device)
        c_0 = torch.randn((self.feature_num_layers, feature_gen_input.size(0), self.feature_num_units)).to(self.device)
        # feature generator
        feature_rnn_output, _ = self.feature_rnn(feature_gen_input, (h_o, c_0))
        # feature_rnn_output, _ = self.feature_rnn(feature_gen_input)
        features = torch.zeros((feature_rnn_output.size(0), feature_rnn_output.size(1), 0)).to(self.device)
        for feature_output_layer in self.feature_output_layers:
            sub_output = feature_output_layer(feature_rnn_output)
            features = torch.cat((features, sub_output), dim=2)

        features = torch.reshape(features, (features.shape[0],
                                            int((features.shape[1] *
                                                 features.shape[
                                                     2]) / self.feature_dim), self.feature_dim))
        return all_attribute, features


# class PositionalEncoding(nn.Module):
#
#     def __init__(self, d_model, max_len=5000):
#         super(PositionalEncoding, self).__init__()
#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
#         if d_model % 2 != 0:
#             div_term_cos = torch.exp(torch.arange(0, d_model - 1, 2).float() * (-math.log(10000.0) / d_model))
#         else:
#             div_term_cos = div_term
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term_cos)
#         pe = pe.unsqueeze(0).transpose(0, 1)
#         self.register_buffer('pe', pe)
#
#     def forward(self, x):
#         x = x + self.pe[:x.size(0), :]
#         return x
#

#
# class ScaledDotProductAttention(nn.Module):
#     ''' Scaled Dot-Product Attention '''
#
#     def __init__(self, temperature, attn_dropout=0.1):
#         super().__init__()
#         self.temperature = temperature
#
#     def forward(self, q, k, v, mask=None):
#         attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
#
#         if mask is not None:
#             attn = attn.masked_fill(mask == 0, -1e9)
#         output = torch.matmul(attn, v)
#         return output, attn
#
#
# class MultiHeadAttention2(nn.Module):
#     ''' Multi-Head Attention module '''
#
#     def __init__(self, d_model, d_k, d_v, n_head=1):
#         super().__init__()
#
#         self.n_head = n_head
#         self.d_k = d_k
#         self.d_v = d_v
#
#         self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
#         self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
#         self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
#
#         self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)
#
#     def forward(self, x, mask=None):
#         d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
#         sz_b, len_q, len_k, len_v = x.size(0), x.size(1), x.size(1), x.size(1)
#         # Pass through the pre-attention projection: b x lq x (n*dv)
#         # Separate different heads: b x lq x n x dv
#         q = self.w_qs(x).view(sz_b, len_q, n_head, d_k)
#         k = self.w_ks(x).view(sz_b, len_k, n_head, d_k)
#         v = self.w_vs(x).view(sz_b, len_v, n_head, d_v)
#
#         # Transpose for attention dot product: b x n x lq x dv
#         q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
#
#         if mask is not None:
#             mask = mask.unsqueeze(1)  # For head axis broadcasting.
#
#         q, attn = self.attention(q, k, v, mask=mask)
#
#         # Transpose to move the head dimension back: b x lq x n x dv
#         # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
#         q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
#         return q, attn
#
#
# class NonDynamicallyQuantizableLinear(nn.Linear):
#     def __init__(self, in_features: int, out_features: int, bias: bool = True,
#                  device=None, dtype=None) -> None:
#         super().__init__(in_features, out_features, bias=bias,
#                          device=device, dtype=dtype)
#
#
# class MultiheadAttention(nn.Module):
#
#     # __constants__ = ['batch_first']
#     # bias_k: Optional[torch.Tensor]
#     # bias_v: Optional[torch.Tensor]
#
#     def __init__(self, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False,
#                  kdim=None, vdim=None, qdim=None, batch_first=False, device=None, dtype=None) -> None:
#         factory_kwargs = {'device': device, 'dtype': dtype}
#         super(MultiheadAttention, self).__init__()
#         self.embed_dim = embed_dim
#         self.kdim = kdim if kdim is not None else embed_dim
#         self.vdim = vdim if vdim is not None else embed_dim
#         self.qdim = qdim if qdim is not None else embed_dim
#         self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim
#
#         self.num_heads = num_heads
#         self.dropout = dropout
#         self.batch_first = batch_first
#         self.head_dim = embed_dim // num_heads
#         assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
#
#         if self._qkv_same_embed_dim is False:
#             self.q_proj_weight = Parameter(torch.empty((embed_dim, self.qdim), **factory_kwargs))
#             self.k_proj_weight = Parameter(torch.empty((embed_dim, self.kdim), **factory_kwargs))
#             self.v_proj_weight = Parameter(torch.empty((embed_dim, self.vdim), **factory_kwargs))
#             self.register_parameter('in_proj_weight', None)
#         else:
#             self.in_proj_weight = Parameter(torch.empty((3 * embed_dim, embed_dim), **factory_kwargs))
#             self.register_parameter('q_proj_weight', None)
#             self.register_parameter('k_proj_weight', None)
#             self.register_parameter('v_proj_weight', None)
#
#         if bias:
#             self.in_proj_bias = Parameter(torch.empty(3 * embed_dim, **factory_kwargs))
#         else:
#             self.register_parameter('in_proj_bias', None)
#         self.out_proj = NonDynamicallyQuantizableLinear(embed_dim, embed_dim, bias=bias, **factory_kwargs)
#
#         if add_bias_kv:
#             self.bias_k = Parameter(torch.empty((1, 1, embed_dim), **factory_kwargs))
#             self.bias_v = Parameter(torch.empty((1, 1, embed_dim), **factory_kwargs))
#         else:
#             self.bias_k = self.bias_v = None
#
#         self.add_zero_attn = add_zero_attn
#
#         self._reset_parameters()
#
#     def _reset_parameters(self):
#         if self._qkv_same_embed_dim:
#             nn.init.xavier_uniform_(self.in_proj_weight)
#         else:
#             nn.init.xavier_uniform_(self.q_proj_weight)
#             nn.init.xavier_uniform_(self.k_proj_weight)
#             nn.init.xavier_uniform_(self.v_proj_weight)
#
#         if self.in_proj_bias is not None:
#             nn.init.constant_(self.in_proj_bias, 0.)
#             nn.init.constant_(self.out_proj.bias, 0.)
#         if self.bias_k is not None:
#             nn.init.xavier_normal_(self.bias_k)
#         if self.bias_v is not None:
#             nn.init.xavier_normal_(self.bias_v)
#
#     def __setstate__(self, state):
#         # Support loading old MultiheadAttention checkpoints generated by v1.1.0
#         if '_qkv_same_embed_dim' not in state:
#             state['_qkv_same_embed_dim'] = True
#
#         super(MultiheadAttention, self).__setstate__(state)
#
#     def forward(self, query, key, value, key_padding_mask=None,
#                 need_weights: bool = True, attn_mask=None):
#
#         if self.batch_first:
#             query, key, value = [x.transpose(1, 0) for x in (query, key, value)]
#
#         if not self._qkv_same_embed_dim:
#             attn_output, attn_output_weights = F.multi_head_attention_forward(
#                 query, key, value, self.embed_dim, self.num_heads,
#                 self.in_proj_weight, self.in_proj_bias,
#                 self.bias_k, self.bias_v, self.add_zero_attn,
#                 self.dropout, self.out_proj.weight, self.out_proj.bias,
#                 training=self.training,
#                 key_padding_mask=key_padding_mask, need_weights=need_weights,
#                 attn_mask=attn_mask, use_separate_proj_weight=True,
#                 q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight,
#                 v_proj_weight=self.v_proj_weight)
#         else:
#             attn_output, attn_output_weights = F.multi_head_attention_forward(
#                 query, key, value, self.embed_dim, self.num_heads,
#                 self.in_proj_weight, self.in_proj_bias,
#                 self.bias_k, self.bias_v, self.add_zero_attn,
#                 self.dropout, self.out_proj.weight, self.out_proj.bias,
#                 training=self.training,
#                 key_padding_mask=key_padding_mask, need_weights=need_weights,
#                 attn_mask=attn_mask)
#         if self.batch_first:
#             return attn_output.transpose(1, 0), attn_output_weights
#         else:
#             return attn_output, attn_output_weights
#
#
# class MultiheadAttention_own(nn.Module):
#     def __init__(self, input_dim, attn_dim, device, num_heads=1):
#         super(MultiheadAttention, self).__init__()
#         self.device = device
#         self.input_dim = input_dim
#         self.attn_dim = attn_dim
#         self.num_heads = num_heads
#         self.q_proj_weight = Parameter(torch.empty((input_dim, attn_dim)), requires_grad=True)
#         self.k_proj_weight = Parameter(torch.empty((input_dim, attn_dim)), requires_grad=True)
#         self.v_proj_weight = Parameter(torch.empty((input_dim, attn_dim)), requires_grad=True)
#         self.softmax = nn.Softmax(dim=2)
#         for name, W in self.named_parameters():
#             if len(W.shape) > 1:
#                 nn.init.xavier_uniform_(W)
#
#     def forward(self, x, attn_mask=None):
#         q = x.matmul(self.q_proj_weight)
#         k = x.matmul(self.k_proj_weight)
#         v = x.matmul(self.v_proj_weight)
#         k = k.transpose(1, 2)
#         scores = torch.bmm(q, k)
#         if attn_mask is not None:
#             scores = scores.masked_fill(attn_mask == 0, -1e9)
#         scores = self.softmax(scores / math.sqrt(self.attn_dim))
#         result = torch.bmm(scores, v)
#         return result
#

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.0, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        if d_model % 2 != 0:
            div_term_cos = torch.exp(torch.arange(0, d_model - 1, 2).float() * (-math.log(10000.0) / d_model))
        else:
            div_term_cos = div_term
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term_cos)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class DoppelGANgerGeneratorAttention(DoppelGANgerGenerator):
    def __init__(self, noise_dim, feature_outputs, attribute_outputs, real_attribute_mask, device, sample_len,
                 attribute_num_units=100, attribute_num_layers=3, num_heads=8, attn_dim=512, attn_mask=True,
                 scope_name="DoppelGANgerGenerator",
                 *args, **kwargs):
        super().__init__(noise_dim, attribute_outputs, real_attribute_mask, device, attribute_num_units,
                         attribute_num_layers,
                         scope_name)
        self.attn_mask = attn_mask
        self.feature_dim = 0
        for feature in feature_outputs:
            self.feature_dim += feature.dim
        embed_dim = noise_dim + self.real_attribute_dim + self.addi_attribute_dim
        # add positional encoding
        self.positional_encoding_0 = PositionalEncoding(d_model=embed_dim, max_len=1000)
        # self.positional_encoding_1 = PositionalEncoding(d_model=attn_dim, max_len=1000)
        self.decoder = torch.nn.TransformerDecoderLayer(d_model=attn_dim, nhead=num_heads, dim_feedforward=attn_dim,
                                                        batch_first=True)
        # create multihead self attention
        # initialize weights
        self.feature_output_layers = nn.ModuleList()
        feature_counter = 0
        feature_len = len(feature_outputs)
        for i in range(len(feature_outputs) * sample_len):
            modules = [nn.Linear(attn_dim, feature_outputs[feature_counter].dim)]
            if feature_outputs[feature_counter].type_ == OutputType.DISCRETE:
                modules.append(nn.Softmax(dim=-1))
            else:
                if feature_outputs[feature_counter].normalization == Normalization.ZERO_ONE:
                    modules.append(nn.Sigmoid())
                else:
                    modules.append(nn.Tanh())
            feature_counter += 1
            if feature_counter % feature_len == 0:
                feature_counter = 0
            self.feature_output_layers.append(nn.Sequential(*modules))
        # initialize weights
        self.feature_output_layers.apply(init_weights)

    @staticmethod
    def generate_square_subsequent_mask(sz: int, bool_mask=False):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        if not bool_mask:
            mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, real_attribute_noise, addi_attribute_noise, feature_input_noise):
        all_attribute, feature_gen_input = super().forward(real_attribute_noise, addi_attribute_noise,
                                                           feature_input_noise)
        feature_gen_input = feature_gen_input.transpose(1, 0)
        feature_gen_input = self.positional_encoding_0(feature_gen_input)
        feature_gen_input = feature_gen_input.transpose(1, 0)

        if self.attn_mask:
            attn_mask = self.generate_square_subsequent_mask(feature_gen_input.shape[1]).to(self.device)
        else:
            attn_mask = None
        multihead_attn_output = self.decoder(feature_gen_input, feature_gen_input, tgt_mask=attn_mask,
                                             memory_mask=attn_mask)
        # check output for correlation
        features = torch.zeros((multihead_attn_output.size(0), multihead_attn_output.size(1), 0)).to(self.device)
        for feature_output_layer in self.feature_output_layers:
            sub_output = feature_output_layer(multihead_attn_output)
            features = torch.cat((features, sub_output), dim=2)

        features = torch.reshape(features, (features.shape[0],
                                            int((features.shape[1] *
                                                 features.shape[
                                                     2]) / self.feature_dim), self.feature_dim))
        return all_attribute, features
