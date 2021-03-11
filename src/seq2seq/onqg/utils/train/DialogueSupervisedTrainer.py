import os
import time
import math
import logging
from tqdm import tqdm
import numpy as np
import collections

import torch
from torch import cuda

import onqg.dataset.Constants as Constants
from onqg.dataset.data_processor import preprocess_dialogue_batch


def record_log(logfile, step, loss, ppl, accu, bleu='unk', bad_cnt=0, lr='unk'):
    accu = '{f:3.8f}%/{b:3.8f}%'.format(f=accu[0], b=accu[1])
    bleu = '{f:3.5f}/{b:3.5f}'.format(f=bleu[0], b=bleu[1]) if len(bleu) == 2 else 'unk'
    
    with open(logfile, 'a', encoding='utf-8') as f:
        f.write(str(step) + ':\tloss=' + str(round(loss, 8)) + ',\tppl=' + str(round(ppl, 8)))
        f.write(',\tbleu=' + bleu + ',\taccu=' + accu)
        f.write(',\tbad_cnt=' + str(bad_cnt) + ',\tlr=' + str(lr) + '\n')


class DialogueSupervisedTrainer(object):

    def __init__(self, model, loss, optimizer, forward_translator, backward_translator, 
                 logger, opt, training_data, validation_data):
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.forward_translator = forward_translator
        self.backward_translator = backward_translator
        self.logger = logger
        self.opt = opt

        self.training_data = training_data
        self.validation_data = validation_data

        self.separate = False
        self.answer = False
        self.sep_id = Constants.SEP

        self.is_attn_mask = True if opt.defined_slf_attn_mask else False
        
        self.cntBatch, self.best_ppl, self.best_bleu = 0, math.exp(100), 0

    def cal_performance(self, loss_input):
        '''
        forward: pred, gold, copy_pred, copy_gate, copy_gold, copy_switch, coverage_pred
        backward: ...
        '''
        loss = 0
        if self.opt.mode in ['initialization', 'forward']:
            loss += self.loss.cal_loss(loss_input['forward'])
        if self.opt.mode in ['initialization', 'backward']:
            loss += self.loss.cal_loss(loss_input['backward'])

        f_n_correct, b_n_correct = None, None
        if self.opt.mode in ['initialization', 'forward']:
            ##=== forward ===##
            f_gold, f_pred = loss_input['forward']['gold'], loss_input['forward']['pred']

            f_pred = f_pred.contiguous().view(-1, f_pred.size(2)).max(1)[1]
            f_gold = f_gold.contiguous().view(-1)

            f_n_correct = f_pred.eq(f_gold)
            f_n_correct = f_n_correct.masked_select(f_gold.ne(Constants.PAD)).sum().item()
        if self.opt.mode in ['initialization', 'backward']:
            ##=== backward ===##
            b_gold, b_pred = loss_input['backward']['gold'], loss_input['backward']['pred']

            b_pred = b_pred.contiguous().view(-1, b_pred.size(2)).max(1)[1]
            b_gold = b_gold.contiguous().view(-1)

            b_n_correct = b_pred.eq(b_gold)
            b_n_correct = b_n_correct.masked_select(b_gold.ne(Constants.PAD)).sum().item()

        return loss, (f_n_correct, b_n_correct)

    def save_model(self, better, bleu):
        model_state_dict = self.model.module.state_dict() if len(self.opt.gpus) > 1 else self.model.state_dict()
        model_state_dict = collections.OrderedDict([(x,y.cpu()) for x,y in model_state_dict.items()])
        checkpoint = {
            'model': model_state_dict,
            'settings': self.opt,
            'step': self.cntBatch}

        if self.opt.save_mode == 'all':
            model_name = self.opt.save_model + '_ppl_{ppl:2.5f}.chkpt'.format(ppl=self.best_ppl)
            torch.save(checkpoint, model_name)
        elif self.opt.save_mode == 'best':
            model_name = self.opt.save_model + '.chkpt'
            if better:
                torch.save(checkpoint, model_name)
                print('    - [Info] The checkpoint file has been updated.')
        
        if len(bleu) == 2 and bleu[0] + bleu[1] > self.best_bleu:
            self.best_bleu = bleu[0] + bleu[1]
            model_name = self.opt.save_model + '_' + str(round(bleu[0] * 100, 5)) + '_' + str(round(bleu[1] * 100, 5)) + '_bleu4.chkpt'
            torch.save(checkpoint, model_name)

    def eval_step(self, device, epoch):
        ''' Epoch operation in evaluation phase '''
        self.model.eval()        

        with torch.no_grad():
            max_length, total_loss = 0, 0
            n_word_total_f, n_word_correct_f, n_word_total_b, n_word_correct_b = 0, 0, 0, 0
            for idx in tqdm(range(len(self.validation_data)), mininterval=2, desc='  - (Validation) ', leave=False):
                batch = self.validation_data[idx]
                inputs, gold, fcopy, bcopy = preprocess_dialogue_batch(batch, enc_rnn=self.opt.enc_rnn != '', 
                                                                       dec_rnn=self.opt.dec_rnn != '', copy=self.opt.copy, 
                                                                       attn_mask=self.is_attn_mask, device=device)
                
                ##### ==================== forward ==================== #####
                f_rst, b_rst = self.model(inputs, mode=self.opt.mode)
                ##### ==================== backward ==================== #####
                loss_input = {}

                if self.opt.mode in ['initialization', 'forward']:
                    loss_input['forward'] = {'pred': f_rst['pred'], 'gold': gold['forward']}
                    if self.opt.copy:
                        loss_input['forward']['copy_pred'], loss_input['forward']['copy_gate'] = f_rst['copy_pred'], f_rst['copy_gate']
                        loss_input['forward']['copy_gold'], loss_input['forward']['copy_switch'] = fcopy[0], fcopy[1]
                    if self.opt.coverage:
                        loss_input['forward']['coverage_pred'] = f_rst['coverage_pred']

                if self.opt.mode in ['initialization', 'backward']:
                    loss_input['backward'] = {'pred': b_rst['pred'], 'gold': gold['backward']}
                    if self.opt.copy:
                        loss_input['backward']['copy_pred'], loss_input['backward']['copy_gate'] = b_rst['copy_pred'], b_rst['copy_gate']
                        loss_input['backward']['copy_gold'], loss_input['backward']['copy_switch'] = bcopy[0], bcopy[1]
                    if self.opt.coverage:
                        loss_input['backward']['coverage_pred'] = b_rst['coverage_pred']

                loss, n_correct = self.cal_performance(loss_input)

                total_loss += loss.item()

                if self.opt.mode in ['initialization', 'forward']:
                    n_word_f = gold['forward'].ne(Constants.PAD).sum().item()
                    n_word_total_f += n_word_f
                    n_word_correct_f += n_correct[0]
                if self.opt.mode in ['initialization', 'backward']:
                    n_word_b = gold['backward'].ne(Constants.PAD).sum().item()
                    n_word_total_b += n_word_b
                    n_word_correct_b += n_correct[1]
            
            loss_per_word = total_loss / (n_word_total_f + n_word_total_b)
            accuracy_f = n_word_correct_f / n_word_total_f * 100 if self.opt.mode in ['initialization', 'forward'] else -1
            accuracy_b = n_word_correct_b / n_word_total_b * 100 if self.opt.mode in ['initialization', 'backward'] else -1
            bleu = 'unk'
            perplexity = math.exp(min(loss_per_word, 16))

            if perplexity <= self.opt.translate_ppl:
                if self.cntBatch % self.opt.translate_steps == 0: 
                    bleu_f = self.forward_translator.eval_all(self.model, self.validation_data) if self.opt.mode in ['initialization', 'forward'] else -1
                    bleu_b = self.backward_translator.eval_all(self.model, self.validation_data) if self.opt.mode in ['initialization', 'backward'] else -1
                    bleu = (bleu_f, bleu_b)

        return loss_per_word, (accuracy_f, accuracy_b), bleu

    def train_epoch(self, device, epoch):
        ''' Epoch operation in training phase'''
        if self.opt.extra_shuffle and epoch > self.opt.curriculum:
            self.logger.info('Shuffling...')
            self.training_data.shuffle()

        self.model.train()

        total_loss, report_total_loss = 0, 0
        n_word_total_f, n_word_correct_f, n_word_total_b, n_word_correct_b = 0, 0, 0, 0
        report_n_word_total_f, report_n_word_correct_f, report_n_word_total_b, report_n_word_correct_b = 0, 0, 0, 0

        batch_order = torch.randperm(len(self.training_data))

        for idx in tqdm(range(len(self.training_data)), mininterval=2, desc='  - (Training)   ', leave=False):

            batch_idx = batch_order[idx] if epoch > self.opt.curriculum else idx
            batch = self.training_data[batch_idx]

            ##### ==================== prepare data ==================== #####
            inputs, gold, fcopy, bcopy = preprocess_dialogue_batch(batch, enc_rnn=self.opt.enc_rnn != '', 
                                                                   dec_rnn=self.opt.dec_rnn != '', copy=self.opt.copy, 
                                                                   attn_mask=self.is_attn_mask, device=device)
            
            ##### ==================== forward ==================== #####
            self.model.zero_grad()
            self.optimizer.zero_grad()
            
            f_rst, b_rst = self.model(inputs, mode=self.opt.mode)

            ##### ==================== backward ==================== #####
            loss_input = {}

            if self.opt.mode in ['initialization', 'forward']:
                loss_input['forward'] = {'pred': f_rst['pred'], 'gold': gold['forward']}
                if self.opt.copy:
                    loss_input['forward']['copy_pred'], loss_input['forward']['copy_gate'] = f_rst['copy_pred'], f_rst['copy_gate']
                    loss_input['forward']['copy_gold'], loss_input['forward']['copy_switch'] = fcopy[0], fcopy[1]
                if self.opt.coverage:
                    loss_input['forward']['coverage_pred'] = f_rst['coverage_pred']

            if self.opt.mode in ['initialization', 'backward']:
                loss_input['backward'] = {'pred': b_rst['pred'], 'gold': gold['backward']}
                if self.opt.copy:
                    loss_input['backward']['copy_pred'], loss_input['backward']['copy_gate'] = b_rst['copy_pred'], b_rst['copy_gate']
                    loss_input['backward']['copy_gold'], loss_input['backward']['copy_switch'] = bcopy[0], bcopy[1]
                if self.opt.coverage:
                    loss_input['backward']['coverage_pred'] = b_rst['coverage_pred']

            loss, n_correct = self.cal_performance(loss_input)
            if len(self.opt.gpus) > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.

            if math.isnan(loss.item()) or loss.item() > 1e20:
                print('catch NaN')
                import ipdb; ipdb.set_trace()

            self.optimizer.backward(loss)
            self.optimizer.step()

            ##### ==================== note for epoch report & step report ==================== #####
            total_loss += loss.item()
            report_total_loss += loss.item()

            if self.opt.mode in ['initialization', 'forward']:
                n_word_f = gold['forward'].ne(Constants.PAD).sum().item()
                n_word_total_f += n_word_f
                n_word_correct_f += n_correct[0]
                report_n_word_total_f += n_word_f
                report_n_word_correct_f += n_correct[0]
            
            if self.opt.mode in ['initialization', 'backward']:
                n_word_b = gold['backward'].ne(Constants.PAD).sum().item()
                n_word_total_b += n_word_b
                n_word_correct_b += n_correct[1]
                report_n_word_total_b += n_word_b
                report_n_word_correct_b += n_correct[1]
            
            ##### ==================== evaluation ==================== #####
            self.cntBatch += 1
            if self.cntBatch % self.opt.valid_steps == 0:                
                ### ========== evaluation on dev ========== ###
                valid_loss, valid_accu, valid_bleu = self.eval_step(device, epoch)
                valid_ppl = math.exp(min(valid_loss, 16))

                report_avg_loss = report_total_loss / (report_n_word_total_f + report_n_word_total_b)
                report_avg_ppl = math.exp(min(report_avg_loss, 16))
                report_avg_accu_f = report_n_word_correct_f / report_n_word_total_f * 100 if self.opt.mode in ['initialization', 'forward'] else -1
                report_avg_accu_b = report_n_word_correct_b / report_n_word_total_b * 100 if self.opt.mode in ['initialization', 'backward'] else -1
                
                better = False
                if valid_ppl <= self.best_ppl:
                    self.best_ppl = valid_ppl
                    better = True

                report_total_loss = 0
                report_n_word_total_f, report_n_word_correct_f = 0, 0
                report_n_word_total_b, report_n_word_correct_b = 0, 0
                
                ### ========== update learning rate ========== ###
                self.optimizer.update_learning_rate(better)

                record_log(self.opt.logfile_train, step=self.cntBatch, loss=report_avg_loss, ppl=report_avg_ppl, 
                           accu=(report_avg_accu_f, report_avg_accu_b), bad_cnt=self.optimizer._bad_cnt, lr=self.optimizer._learning_rate)
                record_log(self.opt.logfile_dev, step=self.cntBatch, loss=valid_loss, ppl=math.exp(min(valid_loss, 16)), 
                           accu=valid_accu, bleu=valid_bleu, bad_cnt=self.optimizer._bad_cnt, lr=self.optimizer._learning_rate)

                if self.opt.save_model:
                    self.save_model(better, valid_bleu)

                self.model.train()

        loss_per_word = total_loss / (n_word_total_f + n_word_total_b)
        accuracy_f = n_word_correct_f / n_word_total_f * 100 if self.opt.mode in ['initialization', 'forward'] else -1
        accuracy_b = n_word_correct_b / n_word_total_b * 100 if self.opt.mode in ['initialization', 'backward'] else -1

        return math.exp(min(loss_per_word, 16)), (accuracy_f, accuracy_b)

    def train(self, device):
        ''' Start training '''
        self.logger.info(self.model)

        for epoch_i in range(self.opt.epoch):
            self.logger.info('')
            self.logger.info(' *  [ Epoch {0} ]:   '.format(epoch_i))
            start = time.time()
            ppl, accu = self.train_epoch(device, epoch_i + 1) 

            self.logger.info(' *  - (Training)   ppl: {ppl: 8.5f}, accuracy: (forward) {accuf:3.3f}% (backward) {accub:3.3f}%'.format(ppl=ppl, accuf=accu[0], accub=accu[1]))
            print('                ' + str(time.time() - start) + ' seconds for epoch ' + str(epoch_i))
        