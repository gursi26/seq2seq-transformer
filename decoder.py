from torch import nn
from attention import MultiHeadAttention
from other_modules import FeedForward, PositionalEncoder
import torch, math


class TransformerDecoderBlock(nn.Module):

    def __init__(self, input_dim, num_heads, p=0.1):
        super(TransformerDecoderBlock, self).__init__()
        self.masked_mha = MultiHeadAttention(input_dim, input_dim, num_heads)
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.dropout1 = nn.Dropout(p=p)
        self.encoder_mha = MultiHeadAttention(input_dim, input_dim, num_heads, encoder_kv=True)
        self.layer_norm2 = nn.LayerNorm(input_dim)
        self.dropout2 = nn.Dropout(p=p)
        self.feed_forward = FeedForward(input_dim)
        self.layer_norm3 = nn.LayerNorm(input_dim)
        self.dropout3 = nn.Dropout(p=p)

    def forward(self, x, enc_outputs, dec_mask, enc_mask):
        skip_x = x
        x = self.masked_mha(x, mask=dec_mask)
        x = self.layer_norm1(self.dropout1(x) + skip_x)
        skip_x = x
        x = self.encoder_mha(x, enc_outputs=enc_outputs, mask=enc_mask)
        x = self.layer_norm2(self.dropout2(x) + skip_x)
        skip_x = x
        x = self.feed_forward(x)
        x = self.layer_norm3(self.dropout3(x) + skip_x)
        return x


class TransformerDecoder(nn.Module):

    def __init__(self, input_dim, d_model, num_heads, n_layers, max_seq_len=4096):
        super(TransformerDecoder, self).__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(input_dim, d_model)
        self.pos_enc = PositionalEncoder(d_model, max_seq_len)
        self.decoder_layers = nn.ModuleList([TransformerDecoderBlock(d_model, num_heads) for _ in range(n_layers)])
        self.out_proj = nn.Linear(d_model, input_dim, bias=False)
        self.out_proj.weight = self.embed.weight

    def forward(self, x, enc_outputs, dec_mask, enc_mask):
        x = self.pos_enc(self.embed(x) * math.sqrt(self.d_model))
        for layer in self.decoder_layers:
            x = layer(x=x, enc_outputs=enc_outputs, dec_mask=dec_mask, enc_mask=enc_mask)
        return self.out_proj(x)


def test_transformer_decoder():
    decoder = TransformerDecoder(5000, 512, 8, 5).to("mps")
    x = torch.arange(0, 100).unsqueeze(0).repeat(32, 1).to("mps")
    mask = torch.ones(32, 8, 100, 100).type(torch.bool).to("mps")
    enc_out = torch.randn(32, 1048, 512).to("mps")
    enc_mask = torch.ones(32, 8, 1, 1048).type(torch.bool).to("mps")
    out = decoder(x, enc_out, mask, enc_mask)
    print(out.shape)


if __name__ == "__main__":
    test_transformer_decoder()