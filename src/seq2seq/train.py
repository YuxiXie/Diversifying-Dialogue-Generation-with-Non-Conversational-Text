import os
import xargs
import argparse

import math
import time
import logging
from tqdm import tqdm

import torch
import torch.nn as nn
from torch import cuda

import onqg.dataset.Constants as Constants
from onqg.dataset.Dataset import DialogueDataset
from onqg.dataset.Vocab import Vocab

from onqg.utils.model_builder import build_dialogue_model
from onqg.utils.train.Loss import NLLLoss
from onqg.utils.train.Optim import Optimizer
from onqg.utils.train.DialogueSupervisedTrainer import DialogueSupervisedTrainer
from onqg.utils.translate.DialogueTranslator import DialogueTranslator


def main(opt, logger):
    logger.info('My PID is {0}'.format(os.getpid()))
    logger.info('PyTorch version: {0}'.format(str(torch.__version__)))
    logger.info(opt)

    if torch.cuda.is_available() and not opt.gpus:
        logger.info("WARNING: You have a CUDA device, so you should probably run with -gpus 0")
    if opt.seed > 0:
        torch.manual_seed(opt.seed)
    if opt.gpus:
        if opt.cuda_seed > 0:
            torch.cuda.manual_seed(opt.cuda_seed)
        # cuda.set_device(opt.gpus[0])
    logger.info('My seed is {0}'.format(torch.initial_seed()))
    logger.info('My cuda seed is {0}'.format(torch.cuda.initial_seed()))
    
    ###### ==================== Loading Dataset ==================== ######
    data = torch.load(opt.data)
    vocabularies = data['dict']
    if isinstance(vocabularies['src'], str):
        assert vocabularies['src'] == opt.pretrained
        options = {'transf':True, 'separate':False, 'tgt':False}
        vocabularies['src'] = Vocab.from_opt(pretrained=opt.pretrained, opt=options)
    train_data, valid_data = data['train'], data['valid']

    ### ===== load pre-trained vocabulary ===== ###
    if opt.pre_trained_vocab:
        if not opt.pretrained:
            opt.pre_trained_src_emb = vocabularies['pre-trained']['src']
        opt.pre_trained_tgt_emb = vocabularies['pre-trained']['tgt']
    
    ### ===== wrap datasets ===== ###
    attn_mask_file = '' if not opt.defined_slf_attn_mask else opt.defined_slf_attn_mask + '.train.npy'
    pad_id = vocabularies['src'].lookup('<|endoftext|>') if opt.pretrained.count('gpt2') else Constants.PAD
    trainData = DialogueDataset(train_data, opt.batch_size, copy=opt.copy, 
                                attn_mask_file=attn_mask_file, 
                                opt_cuda=opt.gpus, pad=pad_id)
    validData = DialogueDataset(valid_data, opt.eval_batch_size, copy=opt.copy, 
                                attn_mask_file=attn_mask_file,
                                opt_cuda=opt.gpus, pad=pad_id)
    
    opt.src_vocab_size, opt.tgt_vocab_size = vocabularies['src'].size, vocabularies['tgt'].size
    
    logger.info(' * vocabulary size. source = %d; target = %d' % (opt.src_vocab_size, opt.tgt_vocab_size))
    logger.info(' * number of training batches. %d' % len(trainData))
    logger.info(' * maximum batch size. %d' % opt.batch_size)

    ##### =================== Prepare Model =================== #####
    separate = -1
    device = torch.device('cuda:' + str(opt.gpus[0]) if len(opt.gpus) else 'cpu')
    checkpoint = torch.load(opt.checkpoint) if opt.checkpoint else None
    model, parameters_cnt = build_dialogue_model(opt, device, separate=separate, checkpoint=checkpoint)
    logger.info(' * Number of parameters to learn = %d' % parameters_cnt)

    ##### ==================== Prepare Optimizer ==================== #####
    optimizer = Optimizer.from_opt(model, opt)

    ##### ==================== Prepare Loss ==================== #####
    weight = torch.ones(opt.tgt_vocab_size)
    weight[Constants.PAD] = 0
    loss = NLLLoss(opt, weight=weight, size_average=False)
    if opt.gpus:
        cuda.set_device(opt.gpus[0])
        loss.cuda()
        
    ##### ==================== Prepare Translator ==================== #####
    forward_translator = DialogueTranslator(opt, vocabularies['tgt'], data['valid']['tokens'], vocabularies['src'])
    backward_translator = DialogueTranslator(opt, vocabularies['src'], data['valid']['tokens'], vocabularies['tgt'], reverse=True)
    
    ##### ==================== Training ==================== #####
    trainer = DialogueSupervisedTrainer(model, loss, optimizer, 
                                        forward_translator, backward_translator,
                                        logger, opt, trainData, validData)
    trainer.train(device)


if __name__ == '__main__':
    ##### ==================== parse the options ==================== #####
    parser = argparse.ArgumentParser(description='train.py')
    xargs.add_data_options(parser)
    xargs.add_model_options(parser)
    xargs.add_train_options(parser)
    opt = parser.parse_args()

    ##### ==================== prepare the logger ==================== #####
    logging.basicConfig(format='%(asctime)s [%(levelname)s:%(name)s]: %(message)s', level=logging.INFO)
    log_file_name = time.strftime("%Y%m%d-%H%M%S") + '.log.txt'
    if opt.log_home:
        log_file_name = os.path.join(opt.log_home, log_file_name)
    file_handler = logging.FileHandler(log_file_name, encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)-5.5s:%(name)s] %(message)s'))
    logging.root.addHandler(file_handler)
    logger = logging.getLogger(__name__)

    main(opt, logger)
