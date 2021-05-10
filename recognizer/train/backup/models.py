import torch.nn.functional as F
import torch.nn as nn

def count_params(module):
    return sum([p.numel() for p in module.parameters()])

class EncoderDecoderModel(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

        self.deploy_mode = False
        self.print_params()

    def print_params(self):
        print('Encoder:', count_params(self.encoder))
        print('Decoder:', count_params(self.decoder))

    def deploy(self):
        self.deploy_mode = True
        if hasattr(self.encoder, "switch_to_deplay_in_place"):
            self.encoder.switch_to_deplay_in_place()

    def forward(self, src, tgt=None, tgt_lens=None):
        src = src.float() / 255
        src = (src - 0.449) / 0.226

        if(self.deploy_mode):
            return self.predict_for_deploy(src)

        memory = self.encoder(src)              # NCHW

        memory = memory.permute(0, 3, 1, 2)     # NWCH
        memory = memory.mean(dim=-1)            # NWC

        if tgt is None:
            return self.decoder.predict(memory)
        return self.decoder(tgt, memory, tgt_lens=tgt_lens)

    def predict_for_deploy(self, src):
        memory = self.encoder(src)              # NCHW

        memory = memory.permute(0, 3, 1, 2)     # NWCH
        memory = memory.mean(dim=-1)            # NWC

        logits = self.decoder.predict(memory)
        return logits.argmax(-1)
        