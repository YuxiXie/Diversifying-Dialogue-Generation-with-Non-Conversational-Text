import torch
from torch import cuda
import torch.nn as nn
import argparse
from tqdm import tqdm

from onqg.utils.translate.DialogueTranslator import DialogueTranslator
from onqg.dataset.Dataset import DialogueDataset
from onqg.utils.model_builder import build_dialogue_model


def dump(data, filename, bleu):
    filename = filename.rstrip('txt').rstrip('.') + '_{bleu:3.5f}_.txt'.format(bleu=bleu * 100)
    golds, preds, paras = data[0], data[1], data[2]
    with open(filename, 'w', encoding='utf-8') as f:
        for g, p, pa in zip(golds, preds, paras):
            pa = [w for w in pa if w not in ['[PAD]', '[CLS]']]
            f.write('<para>\t' + ' '.join(pa) + '\n')
            f.write('<gold>\t' + ' '.join(g[0]) + '\n')
            f.write('<pred>\t' + ' '.join(p) + '\n')
            f.write('===========================\n')


def main(opt):
    device = torch.device('cuda' if opt.cuda else 'cpu')

    checkpoint = torch.load(opt.model)
    model_opt = checkpoint['options']
    model_opt.gpus = opt.gpus
    model_opt.beam_size, model_opt.batch_size = opt.beam_size, opt.batch_size
    model_opt.mode = opt.mode

    ### Prepare Data ###
    data = torch.load(opt.data)

    src_vocab, tgt_vocab = data['dict']['src'], data['dict']['tgt']
    validData = DialogueDataset(data['valid'], model_opt.batch_size, 
                                copy=model_opt.copy, opt_cuda=model_opt.gpus)
    
    ### Prepare Model ###
    model, _ = build_dialogue_model(opt, device, checkpoint=checkpoint)
    model.eval()

    if opt.mode in ['initialization', 'forward']:
        forward_translator = DialogueTranslator(opt, tgt_vocab, data['valid']['tokens'], src_vocab)
        bleu_f, outputs_f = forward_translator.eval_all(model, validData, output_sent=True)
        print('\n * forward bleu-4', bleu_f, '\n')
        dump(outputs_f, opt.output, bleu_f)

    if opt.mode in ['initialization', 'backward']:
        backward_translator = DialogueTranslator(opt, src_vocab, data['valid']['tokens'], tgt_vocab, reverse=True)
        bleu_b, outputs_b = backward_translator.eval_all(model, validData, output_sent=True)
        print('\n * backward bleu-4', bleu_b, '\n')
        dump(outputs_b, opt.output, bleu_b)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='translate.py')

    parser.add_argument('-model', required=True, help='Path to model .pt file')
    parser.add_argument('-data', required=True, help='Path to data file')
    parser.add_argument('-output', required=True, help='Path to output the predictions')
    parser.add_argument('-beam_size', type=int, default=5)
    parser.add_argument('-batch_size', type=int, default=32)
    parser.add_argument('-gpus', default=[], nargs='+', type=int)

    opt = parser.parse_args()
    opt.cuda = True if opt.gpus else False
    if opt.cuda:
        cuda.set_device(opt.gpus[0])
    
    main(opt)
