from torch import nn
import torch, math
import matplotlib.pyplot as plt
from encoder import TransformerEncoder
from decoder import TransformerDecoder


class Seq2SeqTransformer(nn.Module):

    def __init__(self, src_dim, tgt_dim, d_model, num_heads, enc_layers, dec_layers, warmup_steps=4000, betas=[0.9, 0.98]):
        super(Seq2SeqTransformer, self).__init__()
        self.register_buffer("src_dim", torch.tensor(src_dim, dtype=torch.long))
        self.register_buffer("tgt_dim", torch.tensor(tgt_dim, dtype=torch.long))
        self.register_buffer("d_model", torch.tensor(d_model, dtype=torch.long))
        self.register_buffer("num_heads", torch.tensor(num_heads, dtype=torch.long))
        self.register_buffer("enc_layers", torch.tensor(enc_layers, dtype=torch.long))
        self.register_buffer("dec_layers", torch.tensor(dec_layers, dtype=torch.long))

        # optimizer and scheduler info
        self.register_buffer("train_step", torch.tensor(1, dtype=torch.long))
        self.register_buffer("epoch", torch.tensor(0, dtype=torch.long))
        self.register_buffer("warmup_steps", torch.tensor(warmup_steps, dtype=torch.long))
        self.register_buffer("betas", torch.tensor(betas))
        
        self.encoder = TransformerEncoder(
            input_dim=src_dim,
            d_model=d_model,
            num_heads=num_heads,
            n_layers=enc_layers
        )
        self.decoder = TransformerDecoder(
            input_dim=tgt_dim,
            d_model=d_model,
            num_heads=num_heads,
            n_layers=dec_layers
        )

    # enc_mask and dec_mask have shape [batch_size, num_heads, 1, seq_len]
    def forward(self, enc_input, dec_input, enc_mask, dec_mask):
        enc_outputs = self.encoder(enc_input, enc_mask)
        out = self.decoder(
            x=dec_input,
            enc_outputs=enc_outputs,
            dec_mask=dec_mask,
            enc_mask=enc_mask
        )
        return out


def test_transformer():
    dev = torch.device("mps")
    batch_size = 32
    model = Seq2SeqTransformer(
        src_dim=14433,
        tgt_dim=29071,
        d_model=512,
        num_heads=8,
        enc_layers=3,
        dec_layers=3
    ).to(dev)
    print(sum([p.numel() for p in model.parameters()]))
    src = torch.arange(0, 100).unsqueeze(0).repeat(batch_size, 1).to(dev)
    dec_input = torch.arange(0, 120).unsqueeze(0).repeat(batch_size, 1).to(dev)
    enc_mask = torch.ones(batch_size, 8, 1, 100).type(torch.bool).to(dev)
    dec_mask = torch.ones(batch_size, 8, 1, 120).type(torch.bool).to(dev)
    out = model(enc_input=src, dec_input=dec_input, enc_mask=enc_mask, dec_mask=dec_mask)
    print(out.shape)


if __name__ == "__main__":
    test_transformer()