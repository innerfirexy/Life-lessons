# from run import *
from tqdm import tqdm
import sys


def unit_test():
    corpus.print_info()
    # It matches the corpus statistics here: https://pytorch.org/text/stable/_modules/torchtext/datasets/wikitext2.html

    outputs = tokenizer.encode_batch(["Hello, y'all!", "How are you 😁 ?"])
    print(outputs[0].tokens)
    print(outputs[1].tokens)

    # test iter
    # count = 0
    # for item in train_iter:
    #     print(len(item.strip()))
    #     count += 1
    #     if count >= 5:
    #         break

    # test loader
    count = 0
    for batch in train_loader:
        print(len(batch))

        ids, mask, lengths = batch
        print('mask', mask)
        print('lengths', lengths)

        ones_count = torch.count_nonzero(mask, dim=1)
        print('ones_count', ones_count)

        data, lengths, targets = process_batch(batch)
        # print('data', data)
        # print(len(data))
        # print(type(batch))
        print('ids.shape', ids.shape)
        # print(batch.size())
        count += 1
        if count >= 1:
            break

def test_loader():
    step = 0
    for i in range(3):
        print(f'epoch {i+1}')
        for batch in train_loader:
            step += 1
            # sys.stdout.write(f'\rstep {step}, train_data.pos {train_data.pos()}')
            sys.stdout.write(f'\rstep {step}')
            sys.stdout.flush()
            # print(batch)
            # if step > 0:
            #     break
        print()


def test_collate_batch2():
    import torch
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file('data/position/cleaned_data_final_pad.json')
    tokenizer.enable_padding(pad_id=1, pad_token="[PAD]") 

    def collate_batch(batch):
        batch = [item.strip() for item in batch] # NOTE:  `if len(item.strip())>0` This condition is not safe, it changes the size of the actual batch. Consider moving it to the previous step, i.e., the raw iter in Corpus class.
        encoded_results = tokenizer.encode_batch(batch)

        ids_list, attn_mask_list, lengths_list = [], [], []
        for res in encoded_results:
            ids_list.append(res.ids)
            attn_mask_list.append(res.attention_mask)
            lengths_list.append(res.attention_mask.count(1)) # mask is of 1s and 0s, the count of 1s is the actual length

        ids = torch.tensor(ids_list, dtype=torch.int64)
        attention_mask = torch.tensor(attn_mask_list, dtype=torch.int64)
        lengths = torch.tensor(lengths_list, dtype=torch.int64)

        return ids, attention_mask, lengths

    def collate_batch2(batch):
        """
        PyTorch tutorial for writing custom code for seq2seq NLP tasks: 
        https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
        """
        seq1, seq2 = zip(*[item.strip().split('\t') for item in batch])

        seq1_ids, _, seq1_lengths = collate_batch(seq1)
        seq2_ids, _, seq2_lengths = collate_batch(seq2)
        
        return seq1_ids, seq1_lengths, seq2_ids, seq2_lengths
    
    batch = ['43 43 43 43\tone thing that I', '40\trecommend']
    seq1_ids, seq1_lengths, seq2_ids, seq2_lengths = collate_batch2(batch)

    print('seq1_ids', seq1_ids)
    print('seq1_lengths', seq1_lengths)
    print('seq2_ids', seq2_ids)
    print('seq2_lengths', seq2_lengths) 

    assert(seq1_lengths.shape == seq2_lengths.shape)
    print(type(seq1_lengths.shape == seq2_lengths.shape))


def test_load_pretrained_embs():
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file('data/position/cleaned_data_final_pad.json')
    import torchtext
    import numpy as np

    def load_pretrained_embed(tokenizer):
        tt_embed = torchtext.vocab.GloVe(name="6B", dim=300)
        vocab = tokenizer.get_vocab()
        embed = np.random.randn(len(vocab), 300)
        for word in vocab:
            if vocab[word] in tt_embed.stoi:
                w_id = vocab[word]
                embed[w_id, :] = tt_embed[word]
        return embed

    embed = load_pretrained_embed(tokenizer)
    print(embed.shape)


if __name__ == '__main__':
    # unit_test()
    # test_loader()
    # test_collate_batch2()
    test_load_pretrained_embs()