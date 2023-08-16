import torch
from torch.utils.data import DataLoader
from dataset import EngSpaDataset


class PadCollate:

    def __init__(self, dim=0):
        self.dim = dim

    def pad_tensor_gen_mask(self, vec, pad, dim):
        mask = torch.cat([torch.ones(vec.shape[dim]), torch.zeros(pad - vec.shape[dim])], dim=dim)
        out = torch.cat([vec, torch.zeros(pad - vec.shape[dim]), mask], dim=dim)
        if (pad - vec.shape[dim]) > 0 and out[vec.shape[dim] - 1] != 2:
            out[vec.shape[dim]] = 2
        return out

    def __call__(self, batch):
        max_src_len = max(map(lambda x: x[0].shape[self.dim], batch))
        max_dec_len = max(map(lambda x: x[1].shape[self.dim], batch))
        batch = [(self.pad_tensor_gen_mask(x, max_src_len, 0), self.pad_tensor_gen_mask(y, max_dec_len, 0), self.pad_tensor_gen_mask(z, max_dec_len, 0)) for x, y, z in batch]
        src_and_mask = torch.stack([x[0] for x in batch], dim=0)
        dec_and_mask = torch.stack([x[1] for x in batch], dim=0)
        tgt_and_mask = torch.stack([x[2] for x in batch], dim=0)
        src, src_mask = src_and_mask.chunk(2, dim=-1)
        dec, dec_mask = dec_and_mask.chunk(2, dim=-1)
        tgt, _ = tgt_and_mask.chunk(2, dim=-1)
        return (src.type(torch.long), src_mask.type(torch.bool)), (dec.type(torch.long), dec_mask.type(torch.bool)), tgt.type(torch.long)


# padding_mask has shape [batch_size, seq_len]
def prepare_mask(padding_mask, no_peek_future=False):
    padding_mask = padding_mask.unsqueeze(1).unsqueeze(-2)
    if no_peek_future:
        no_peek_future = torch.tril(torch.ones(padding_mask.shape[-1], padding_mask.shape[-1])).type(torch.bool)
        padding_mask = padding_mask * no_peek_future
    return padding_mask


def inference(src_seq, model, dataset, device, max_gen_len=100):
    src_seq = torch.tensor([dataset.eng2idx[w] for w in dataset.preprocess(src_seq).split()]).unsqueeze(0)
    src_mask = prepare_mask(torch.ones_like(src_seq).type(torch.bool))
    enc_outputs = model.encoder(src_seq.to(device), src_mask.to(device))

    i = 0
    dec_input = torch.tensor([dataset.spa2idx["<SOS>"]])
    while i < max_gen_len and dec_input[-1].item() != dataset.spa2idx["<EOS>"]:
        dec_mask = prepare_mask(torch.ones_like(dec_input.unsqueeze(0)).type(torch.bool), no_peek_future=True)
        out = model.decoder(x=dec_input.unsqueeze(0).to(device), enc_outputs=enc_outputs, enc_mask=src_mask.to(device), dec_mask=dec_mask.to(device)).squeeze(0)[-1]
        out = out.softmax(dim=-1)
        # dec_input = torch.cat([dec_input, out.multinomial(1).to("cpu")], dim=0)
        dec_input = torch.cat([dec_input, out.argmax().view(1).to("cpu")], dim=0)
        i += 1
    return " ".join([dataset.idx2spa[i.item()] for i in dec_input])


class TransformerScheduler:

    def __init__(self, opt, warmup_steps, d_model):
        self.opt = opt
        self.warmup_steps = warmup_steps
        self.d_model = d_model
        self.step_num = 0

    def step(self):
        self.step_num += 1
        for p in self.opt.param_groups:
            p["lr"] = (self.d_model ** (-0.5)) * min(self.step_num ** (-0.5), self.step_num * self.warmup_steps ** (-1.5))


def test_padding():
    loader = DataLoader(EngSpaDataset("eng-spa.csv"), batch_size=32, shuffle=False, collate_fn=PadCollate())
    (src, src_mask), (dec_input, dec_mask), tgt = next(iter(loader))
    print(src[0])
    print(dec_input[0])
    print(tgt[0])
    src_mask = prepare_mask(src_mask)
    dec_mask = prepare_mask(dec_mask, no_peek_future=True)
    print(src_mask.shape, dec_mask.shape)
    print(src_mask[0][0])
    print(dec_mask[0][0])


if __name__ == "__main__":
    test_padding()