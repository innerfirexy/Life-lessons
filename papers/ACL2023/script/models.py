import torch
import math
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class RNN(nn.Module):
    """
    Input is not padded
    """
    def __init__(self, model_type, ntokens, emsize, nhid, nlayers, dropout, args):
        super(RNN, self).__init__()
        self.rnn_type = model_type
        self.ntokens = ntokens # This is the `ntokens` in lstm_gesture.yaml
        self.ninp = emsize
        self.nhid = nhid
        self.nlayers = nlayers
        self.dropout = dropout

        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(self.ntokens, self.ninp)
        if self.rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, self.rnn_type)(self.ninp, self.nhid, self.nlayers, dropout=self.dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[self.rnn_type]
            except KeyError:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(self.ninp, self.nhid, self.nlayers, nonlinearity=nonlinearity, dropout=self.dropout)
        self.decoder = nn.Linear(self.nhid, self.ntokens)
        self.init_weights(args)

    def init_weights(self, args):
        initrange = 0.1
        if args.pretrained_embeds:
            self.encoder.from_pretrained(args.pretrained_embeds)
        else:
            self.encoder.weight.data.uniform_(-initrange, initrange)

        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, x, h):
        emb = self.drop(self.encoder(x)) # dropout applied to embedding layer; shape: [bsize, maxseqlen, embsize], i.e., [L, B, E]
        out, h = self.rnn(emb, h) # out shape: [L, B, H]
        out = self.drop(out)
        decoded = self.decoder(out) # decoded shape: [B, L, V], V is vocabulary size

        return decoded, h

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                    weight.new_zeros(self.nlayers, bsz, self.nhid))
        else:
            return weight.new_zeros(self.nlayers, bsz, self.nhid)


class RNN_padded(nn.Module):
    """
    Input to encoder is padded
    Container module with an encoder, a recurrent module, and a decoder.
    """
    def __init__(self, model_type, ntokens, emsize, nhid, nlayers, args):
        super(RNN_padded, self).__init__()
        self.rnn_type = model_type
        self.ntokens = ntokens
        self.ninp = emsize
        self.nhid = nhid
        self.nlayers = nlayers
        
        self.dropout = args.dropout
        self.drop = nn.Dropout(args.dropout)
        self.encoder = nn.Embedding(self.ntokens, self.ninp)
        if self.rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, self.rnn_type)(self.ninp, self.nhid, self.nlayers, dropout=self.dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[self.rnn_type]
            except KeyError:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(self.ninp, self.nhid, self.nlayers, nonlinearity=nonlinearity, dropout=self.dropout)
        self.decoder = nn.Linear(self.nhid, self.ntokens)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if args.tied:
            if self.nhid != self.ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight
        self.init_weights(args)

    def init_weights(self, args):
        initrange = 0.1
        if args.pretrained_embeds is not None:
            self.encoder.from_pretrained(args.pretrained_embeds)
        else:
            self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, x, x_lengths, h):
        """
        # Source: https://stackoverflow.com/questions/51030782/why-do-we-pack-the-sequences-in-pytorch
        # When training RNN (LSTM or GRU or vanilla-RNN), it is difficult to batch the variable length sequences. For example: if the length of sequences in a size 8 batch is [4,6,8,5,4,3,7,8], you will pad all the sequences and that will result in 8 sequences of length 8. You would end up doing 64 computations (8x8), but you needed to do only 45 computations. Moreover, if you wanted to do something fancy like using a bidirectional-RNN, it would be harder to do batch computations just by padding and you might end up doing more computations than required.
        # Instead, PyTorch allows us to pack the sequence, internally packed sequence is a tuple of two lists. One contains the elements of sequences. Elements are interleaved by time steps (see example below) and other contains the size of each sequence the batch size at each step. This is helpful in recovering the actual sequences as well as telling RNN what is the batch size at each time step. This has been pointed by @Aerin. This can be passed to RNN and it will internally optimize the computations.
        # Here's a code example:
        a = [torch.tensor([1,2,3]), torch.tensor([3,4])]
        b = torch.nn.utils.rnn.pad_sequence(a, batch_first=True)
        >>>>
        tensor([[ 1,  2,  3],
            [ 3,  4,  0]])
        torch.nn.utils.rnn.pack_padded_sequence(b, batch_first=True, lengths=[3,2])
        >>>>PackedSequence(data=tensor([ 1,  3,  2,  4,  3]), batch_sizes=tensor([ 2,  2,  1]))
        """
        emb = self.drop(self.encoder(x)) # dropout applied to embedding layer; shape: [bsize, maxseqlen, embsize], i.e., [L, B, E]
        emb_packed = nn.utils.rnn.pack_padded_sequence(emb, x_lengths, batch_first=False, enforce_sorted=False)
        out, h = self.rnn(emb_packed, h) # out shape: [L, B, H]
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=False)

        out = self.drop(out)
        decoded = self.decoder(out) # decoded shape: [B, L, V], V is vocabulary size

        return decoded, h

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                    weight.new_zeros(self.nlayers, bsz, self.nhid))
        else:
            return weight.new_zeros(self.nlayers, bsz, self.nhid)


class MixedRNN(RNN_padded):
    def __init__(self, args):
        self.g_ntokens = args.g_ntokens
        self.g_emsize = args.g_emsize
        self.w_ntokens = args.w_ntokens
        self.w_emsize = args.w_emsize
        self.mix_type = args.mix_type
        if self.mix_type == 'sum':
            self.mix_emsize = self.w_emsize
        elif self.mix_type == 'concat':
            self.mix_emsize = self.w_emsize + self.g_emsize
        elif self.mix_type == 'bilinear':
            self.mix_emsize = args.mix_emsize
            self.bilinear_encoder = nn.Bilinear(self.w_emsize, self.g_emsize, self.mix_emsize)

        # Use self.mix_emsize to initialize the self.rnn
        super(MixedRNN, self).__init__(args.model_type, self.w_ntokens, self.mix_emsize, args.nhid, args.nlayers, args)
        if self.mix_emsize != self.w_emsize:
            self.encoder = nn.Embedding(self.w_ntokens, self.w_emsize) # override default initialization
        self.g_encoder = nn.Embedding(self.g_ntokens + 1, self.g_emsize, padding_idx=self.g_ntokens) # +1 for the g_pad_id
        self.g2w_encoder = nn.Linear(self.g_emsize, self.w_emsize)
        self.w2g_encoder = nn.Linear(self.w_emsize, self.g_emsize)
        self.nhid = args.nhid

        self.pred_task = args.pred_task
        if self.pred_task == 'word':
            self.decoder = nn.Linear(self.nhid, self.w_ntokens)
        elif self.pred_task == 'gesture':
            self.decoder = nn.Linear(self.nhid, self.g_ntokens)
    
    def forward(self, x1, x2, x_lengths, h):
        w_emb = self.drop(self.encoder(x1))
        g_emb = self.drop(self.g_encoder(x2))
        if self.pred_task == 'word':
            g_emb = self.g2w_encoder(g_emb)
        elif self.pred_task == 'gesture':
            w_emb = self.w2g_encoder(w_emb)

        if self.mix_type == 'sum':
            mix_emb = w_emb + g_emb
        elif self.mix_type == 'concat':
            mix_emb = torch.cat((w_emb, g_emb), dim=-1) # last dim is for embeddings
        elif self.mix_type == 'bilinear':
            mix_emb = self.bilinear_encoder(w_emb, g_emb)
          
        emb_packed = nn.utils.rnn.pack_padded_sequence(mix_emb, x_lengths, batch_first=False, enforce_sorted=False)
        out, h = self.rnn(emb_packed, h) # out shape: [L, B, H]
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=False)
        out = self.drop(out)
        decoded = self.decoder(out) # decoded shape: [L, B, V], V is vocabulary size

        return decoded, h


class PositionalEncoding(nn.Module):
    """
    Adapted from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    with modification to batch_size order
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000, batch_first=False):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.batch_first = batch_first
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        x: Tensor, if batch_first, shape [batch_size, seq_len, embedding_dim], otherwise, shape [seq_len, batch_size, embedding_dim]
        """
        if self.batch_first:
            x = x + self.pe[:, :x.size(0), :]
        else:
            x = x + self.pe[:x.size(0)]
        return self.dropout(x)


###
# Single-modal transformer
###
class TransformerModel(nn.Module):
    def __init__(self, ntokens: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float, args):
        super().__init__()
        self.pos_encoder = PositionalEncoding(d_model, dropout, batch_first=False)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntokens, d_model)
        self.d_model = d_model
        self.decoder = nn.Linear(d_model, ntokens)
        self.drop = nn.Dropout(p=dropout)
        self.init_weights(args)

    def init_weights(self, args) -> None:
        initrange = 0.1
        if args.pretrained_embeds is not None:
            self.encoder.from_pretrained(args.pretrained_embeds)
        else:
            self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, src_mask: Tensor) -> Tensor:
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [seq_len, batch_size, ntokens]
        """
        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output)
        return output


def generate_square_subsequent_mask(sz: int) -> Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)


###
# Transformer model for padded input
###
class TransformerModel_padded(TransformerModel):
    def __init__(self, args):
        super(TransformerModel_padded, self).__init__(args.w_ntokens, args.d_model, args.nhead,
                                                      args.d_hid, args.nlayers, args.dropout, args)
        self.w_ntokens = args.w_ntokens
        self.w_emsize = args.w_emsize

    def forward(self, src: Tensor, src_mask: Tensor, src_key_padding_mask: Tensor) -> Tensor:
        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask, src_key_padding_mask)
        output = self.decoder(output)
        return output


###
# Mix-modal transformer
###
class MixedTransformerModel(TransformerModel):
    def __init__(self, args):
        self.g_ntokens = args.g_ntokens
        self.g_emsize = args.g_emsize
        self.w_ntokens = args.w_ntokens
        self.w_emsize = args.w_emsize

        self.mix_type = args.mix_type
        if self.mix_type == 'sum':
            self.mix_emsize = self.w_emsize
        elif self.mix_type == 'concat':
            self.mix_emsize = self.w_emsize + self.g_emsize
        elif self.mix_type == 'bilinear':
            self.mix_emsize = args.mix_emsize

        # Initialize with self.mix_emsize, instead of d_model
        super(MixedTransformerModel, self).__init__(args.w_ntokens, self.mix_emsize, args.nhead,
                                                    args.d_hid, args.nlayers, args.dropout, args)
        if self.mix_emsize != self.w_emsize:
            self.encoder = nn.Embedding(self.w_ntokens, self.w_emsize) # override default initialization
        if self.mix_type == 'bilinear':
            self.bilinear_encoder = nn.Bilinear(self.w_emsize, self.g_emsize, self.mix_emsize)
        self.g_encoder = nn.Embedding(self.g_ntokens + 1, self.g_emsize, padding_idx=self.g_ntokens) # +1 for the g_pad_id
        self.g2w_encoder = nn.Linear(self.g_emsize, self.w_emsize)
        self.w2g_encoder = nn.Linear(self.w_emsize, self.g_emsize)

        self.pred_task = args.pred_task
        if self.pred_task == 'word':
            self.decoder = nn.Linear(self.d_model, self.w_ntokens)
        elif self.pred_task == 'gesture':
            self.decoder = nn.Linear(self.d_model, self.g_ntokens)


    def forward(self, src: Tensor, src_mask: Tensor, src_key_padding_mask: Tensor, gestures: Tensor) -> Tensor:
        w_emb = self.drop(self.encoder(src))
        g_emb = self.drop(self.g_encoder(gestures))
        if self.pred_task == 'word':
            g_emb = self.g2w_encoder(g_emb)
        elif self.pred_task == 'gesture':
            w_emb = self.w2g_encoder(w_emb)

        if self.mix_type == 'sum':
            mix_emb = w_emb + g_emb
        elif self.mix_type == 'concat':
            mix_emb = torch.cat((w_emb, g_emb), dim=-1) # Last dim is embedding
        elif self.mix_type == 'bilinear':
            mix_emb = self.bilinear_encoder(w_emb, g_emb)

        # Go through transformer encoder
        src = mix_emb * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask, src_key_padding_mask)
        output = self.decoder(output)

        return output