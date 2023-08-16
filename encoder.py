from torch import nn
from attention import MultiHeadAttention
from other_modules import FeedForward, PositionalEncoder
import torch, math


class TransformerEncoderBlock(nn.Module):

    def __init__(self, input_dim, num_heads, p=0.1):
        super(TransformerEncoderBlock, self).__init__()
        self.mha = MultiHeadAttention(input_dim, input_dim, num_heads)
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.dropout1 = nn.Dropout(p=p)
        self.feed_forward = FeedForward(input_dim)
        self.layer_norm2 = nn.LayerNorm(input_dim)
        self.dropout2 = nn.Dropout(p=p)

    def forward(self, x, mask=None):
        skip_x = x
        x = self.mha(x, mask=mask)
        x = self.layer_norm1(self.dropout1(x) + skip_x)
        skip_x = x
        x = self.feed_forward(x)
        return self.layer_norm2(self.dropout2(x) + skip_x)


class TransformerEncoder(nn.Module):

    def __init__(self, input_dim, d_model, num_heads, n_layers, max_seq_len=4096):
        super(TransformerEncoder, self).__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(input_dim, d_model)
        self.pos_enc = PositionalEncoder(d_model, max_seq_len)
        self.encoder_layers = nn.ModuleList([TransformerEncoderBlock(d_model, num_heads) for _ in range(n_layers)])

    def forward(self, x, mask):
        x = self.pos_enc(self.embed(x) * math.sqrt(self.d_model))
        for layer in self.encoder_layers:
            x = layer(x, mask)
        return x


def test_transformer_encoder():
    encoder = TransformerEncoder(5000, 512, 8, 3).to("mps")
    x = torch.arange(0, 100).view(1, 100).repeat(32, 1).to("mps")
    mask = torch.ones(32, 8, 100, 100).type(torch.bool).to("mps")
    out = encoder(x, mask)
    print(out.shape)

if __name__ == "__main__":
    test_transformer_encoder()