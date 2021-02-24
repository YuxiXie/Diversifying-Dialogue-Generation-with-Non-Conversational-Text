import math
import torch
import argparse
import csv
from tqdm import tqdm

import pargs
import onqg.dataset.Constants as Constants
from onqg.dataset.Vocab import Vocab


def load_vocab(filename):
    vocab_dict = {}
    with open(filename, 'r', encoding='utf-8') as f:
        text = f.read().strip().split('\n')
    text = [word.split(' ') for word in text]
    vocab_dict = {word[0]:word[1:] for word in text}
    vocab_dict = {k:[float(d) for d in v] for k,v in vocab_dict.items()}
    return vocab_dict


def convert_word_to_idx(data, vocabularies, opt):

    def lower_sent(sent):
        for idx, w in enumerate(sent):
            if w not in ['[BOS]', '[EOS]', '[PAD]', '[SEP]', '[UNK]', '[CLS]']:
                sent[idx] = w.lower()
        return sent

    indexes, tokens = {}, {}
    for k, v in data.items():
        indexes[k] = [] if v else None
        tokens[k] = [] if v else None
        indexes[k] = [[] for _ in v] if k.count('feats') and v else indexes[k]
        tokens[k] = [[] for _ in v] if k.count('feats') and v else tokens[k]

    for i in tqdm(range(len(data['src'])), desc='     (convert tokens to ids)     '):
        src, tgt = data['src'][i], data['tgt'][i]
        if not opt.bert_tokenizer:
            src = lower_sent(src)
        if not opt.bert_tokenizer or not opt.share_vocab:
            tgt = lower_sent(tgt)
        if opt.bert_tokenizer:
            src = vocabularies['src'].tokenizer.tokenize(' '.join(src))
            if opt.share_vocab:
                tgt = [Constants.CLS_WORD] + vocabularies['tgt'].tokenizer.tokenize(' '.join(tgt)) + [Constants.SEP_WORD]
            else:
                tgt = [Constants.BOS_WORD] + tgt + [Constants.EOS_WORD]
        else:
            tgt = [Constants.BOS_WORD] + tgt + [Constants.EOS_WORD]
        
        if len(src) <= opt.src_seq_length and len(tgt) - 1 <= opt.tgt_seq_length and (len(src) - 8) * (len(tgt) - 5) > 0:
            for key, text in data.items():
                if text:
                    if key.count('feats'):
                        for j in range(len(indexes[key])):
                            indexes[key][j].append(vocabularies[key][j].convertToIdx(text[j][i]))
                    else:
                        text = text[i]
                        text = src if key == 'src' else text
                        text = tgt if key == 'tgt' else text
                    
                        tokens[key].append(text)
                        indexes[key].append(vocabularies[key].convertToIdx(text))

    print("change data size from " + str(len(data['src'])) + " to " + str(len(indexes['src'])))

    return indexes, tokens


def get_embedding(vocab_dict, vocab):

    def get_vector(idx):
        word = vocab.idxToLabel[idx]
        if idx in vocab.special or word not in vocab_dict:
            vector = torch.tensor([])
            vector = vector.new_full((opt.word_vec_size,), 1.0)
            vector.normal_(0, math.sqrt(6 / (1 + vector.size(0))))
        else:
            vector = torch.Tensor(vocab_dict[word])
        return vector
    
    embedding = [get_vector(idx) for idx in range(vocab.size)]
    embedding = torch.stack(embedding)
    
    print(embedding.size())

    return embedding


def get_data(filename):
    rst = {'src': [], 'tgt': []}
    
    with open(filename, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rst['src'].append(row['input_text'])
            rst['tgt'].append(row['target_text'])

    return rst


def merge_ans(src, ans):
    rst = [s + [Constants.SEP_WORD] + a + [Constants.SEP_WORD] for s, a in zip(src, ans)]
    return rst


def wrap_copy_idx(splited, tgt, tgt_vocab, bert, vocab_dict):
    
    def wrap_sent(sp, t):
        sp_dict = {w:idx for idx, w in enumerate(sp)}
        swt, cpt = [0 for w in t], [0 for w in t]
        for i, w in enumerate(t):
            if w in sp_dict and tgt_vocab.frequencies[tgt_vocab.lookup(w)] <= 10:
                swt[i] = 1
                cpt[i] = sp_dict[w]
        return torch.Tensor(swt), torch.LongTensor(cpt)
    
    copy = [wrap_sent(sp, t) for sp, t in zip(splited, tgt)]
    switch, cp_tgt = [c[0] for c in copy], [c[1] for c in copy]
    return {'switch': switch, 'tgt': cp_tgt}
    

def main(opt):
    #========== get data ==========#
    train_file, valid_file = opt.train_file, opt.valid_file
    train_data, valid_data = get_data(train_file), get_data(valid_file)

    #========== build vocabulary ==========#
    vocabularies = {}
    
    pre_trained_vocab = load_vocab(opt.pre_trained_vocab) if opt.pre_trained_vocab else None
    if opt.bert_tokenizer:
        options = {'transf':True, 'separate':False, 'tgt':True}
        bert_tokenizer = Vocab.from_opt(pretrained=opt.bert_tokenizer, opt=options)
    if opt.share_vocab:
        print("build src & tgt vocabulary")
        if opt.bert_tokenizer:
            vocabularies['src'] = vocabularies['tgt'] = bert_tokenizer
        else:
            corpus = train_data['src'] + train_data['tgt']
            options = {'lower':True, 'mode':opt.vocab_trunc_mode, 'transf':True, 'separate':False, 'tgt':True,
                       'size':max(opt.src_vocab_size, opt.tgt_vocab_size),
                       'frequency':min(opt.src_words_min_frequency, opt.tgt_words_min_frequency)}
            vocab = Vocab.from_opt(corpus=corpus, opt=options)
            vocabularies['src'] = vocabularies['tgt'] = vocab
        vocabularies['ans'] = None
    else:
        print("build src vocabulary")
        if opt.bert_tokenizer:
            assert opt.answer != 'enc'
            vocabularies['src'], vocabularies['ans'] = bert_tokenizer, None
        else:
            corpus = train_data['src']
            options = {'lower':True, 'mode':'size', 'transf':opt.answer != 'enc', 'separate':False, 'tgt':False, 
                       'size':opt.src_vocab_size, 'frequency':opt.src_words_min_frequency}
            vocabularies['src'] = Vocab.from_opt(corpus=corpus, opt=options)
            vocabularies['ans'] = None
        print("build tgt vocabulary")
        options = {'lower':True, 'mode':opt.vocab_trunc_mode, 'transf':False, 'separate':False, 'tgt':True, 
                   'size':opt.tgt_vocab_size, 'frequency':opt.tgt_words_min_frequency}
        vocabularies['tgt'] = Vocab.from_opt(corpus=train_data['tgt'], opt=options)
    
    options = {'lower':False, 'mode':'size', 'size':opt.feat_vocab_size, 'frequency':opt.feat_words_min_frequency,
               'transf':True, 'separate':False, 'tgt':False}
    vocabularies['feature'] = None
    vocabularies['ans_feature'] = None
    
    #========== word to index ==========#
    train_indexes, train_tokens = convert_word_to_idx(train_data, vocabularies, opt)
    valid_indexes, valid_tokens = convert_word_to_idx(valid_data, vocabularies, opt)
    vocabularies['tgt'].word_count(train_indexes['tgt'] + valid_indexes['tgt'])

    train_indexes['copy'] = {
        'forward': wrap_copy_idx(train_tokens['src'], train_tokens['tgt'], vocabularies['tgt'], 
                                 opt.bert_tokenizer, pre_trained_vocab) if opt.copy else {'switch': None, 'tgt': None},
        'backward': wrap_copy_idx(train_tokens['tgt'], train_tokens['src'], vocabularies['src'],
                                  opt.bert_tokenizer, pre_trained_vocab) if opt.copy else {'switch': None, 'tgt': None}
    }
    valid_indexes['copy'] = {
        'forward': wrap_copy_idx(valid_tokens['src'], valid_tokens['tgt'], vocabularies['tgt'], 
                                 opt.bert_tokenizer, pre_trained_vocab) if opt.copy else {'switch': None, 'tgt': None},
        'backward': wrap_copy_idx(valid_tokens['tgt'], valid_tokens['src'], vocabularies['src'], 
                                  opt.bert_tokenizer, pre_trained_vocab) if opt.copy else {'switch': None, 'tgt': None}
    }

    #========== prepare pretrained vetors ==========#
    if pre_trained_vocab:
        pre_trained_src_vocab = None if opt.bert_tokenizer else get_embedding(pre_trained_vocab, src_vocab)
        pre_trained_tgt_vocab = None if (opt.bert_tokenizer and opt.share_vocab) else get_embedding(pre_trained_vocab, tgt_vocab)
        pre_trained_ans_vocab = get_embedding(pre_trained_vocab, ans_vocab) if opt.answer == 'enc' else None
        pre_trained_vocab = {'src':pre_trained_src_vocab, 'tgt':pre_trained_tgt_vocab, 'ans':pre_trained_ans_vocab}

    vocabularies['pre-trained'] = pre_trained_vocab

    #========== save data ===========#
    valid_indexes['tokens'] = valid_tokens
    train_indexes['tokens'] = train_tokens
    data = {'settings': opt, 
            'dict': vocabularies,
            'train': train_indexes,
            'valid': valid_indexes
        }
    
    torch.save(data, opt.save_data)
    # import ipdb; ipdb.set_trace()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='preprocess.py')
    pargs.add_options(parser)
    opt = parser.parse_args()
    main(opt)
