from torch import nn
import torch


class QKVLayer(nn.Module):

    def __init__(self, input_dim, output_dim, num_heads, mode="qkv"):
        super(QKVLayer, self).__init__()
        assert output_dim % num_heads == 0    # output_dim must be divisible by num_heads
        assert mode == "qkv" or mode == "kv" or mode == "q"
        self.len_mode = len(mode)
        self.num_heads, self.head_dim = num_heads, output_dim // num_heads
        self.qkv_linear = nn.Linear(input_dim, output_dim * self.len_mode, bias=False)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        qkv = self.qkv_linear(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim * self.len_mode)
        qkv = qkv.permute(0, 2, 1, 3)
        return qkv.chunk(self.len_mode, dim=-1)


class ScaledDotProductAttention(nn.Module):

    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    # mask must be of shape [batch_size, num_heads, seq_len, seq_len]
    # mask must combine no peek mask and padding mask
    def forward(self, q, k, v, mask):
        d_k = k.shape[-1]
        qk = q.matmul(k.transpose(-1, -2)) / d_k
        if mask is not None:
            qk = qk.masked_fill(~mask, -torch.inf)
        attn_weights = qk.softmax(dim=-1)
        return attn_weights.matmul(v)


# if encoder_kv is true, forward call expects x values and outputs from encoder block
# queries generated from x values and k, v generated from encoder outputs to attend to
class MultiHeadAttention(nn.Module):

    def __init__(self, input_dim, output_dim, num_heads, encoder_kv=False) -> None:
        super(MultiHeadAttention, self).__init__()
        self.encoder_kv = encoder_kv
        if not encoder_kv:
            self.qkv_layer = QKVLayer(input_dim, output_dim, num_heads)
        else:
            self.q_layer = QKVLayer(input_dim, output_dim, num_heads, mode="q")
            self.kv_layer = QKVLayer(input_dim, output_dim, num_heads, mode="kv")
        self.attention = ScaledDotProductAttention()
        self.out_proj = nn.Linear(output_dim, output_dim)

    # mask must be of shape [batch_size, num_heads, seq_len, seq_len], combining padding and no peek mask
    # x of shape [batch_size, seq_len, embed_dim]
    # enc_outputs of shape [batch_size, seq_len2, embed_dim]
    def forward(self, x, enc_outputs=None, mask=None):
        batch_size, seq_len, _ = x.shape

        if not self.encoder_kv:
            q, k, v = self.qkv_layer(x)
        else:
            assert enc_outputs is not None
            q = self.q_layer(x)[0]
            k, v = self.kv_layer(enc_outputs)

        attn_outputs = self.attention(q, k, v, mask)
        return self.out_proj(attn_outputs.permute(0, 2, 1, 3).reshape(batch_size, seq_len, -1))


def test_multi_head_attention():
    mha = MultiHeadAttention(512, 512, 8)
    mha_with_enc_att = MultiHeadAttention(512, 512, 8, encoder_kv=True)

    x = torch.randn(32, 100, 512)
    mask = torch.ones(32, 8, 100, 100).type(torch.bool)
    enc_out = torch.randn(32, 1048, 512)

    out1 = mha(x, mask=mask)
    out2 = mha_with_enc_att(out1, enc_outputs=enc_out)
    print(out1.shape)
    print(out2.shape)


if __name__ == "__main__":
    test_multi_head_attention()