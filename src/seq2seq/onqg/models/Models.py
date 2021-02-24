import torch
import torch.nn as nn
from torch.autograd import Variable


import onqg.dataset.Constants as Constants


class Dialoguer(nn.Module):
    '''
    A seq2seq-based Dialogue Generation model
    include both forward & backward generation

    Input: (1) src_seq
           (2) tgt_seq
           (3) max_length: max lengthes of sentences in src / tgt ——> for DataParallel Class

    Output: results output from the 2 decoders (type: list of dict) 
    '''
    def __init__(self, encoder, forward_decoder, backward_decoder):
        super(Dialoguer, self).__init__()

        self.encoder = encoder
        self.forward_decoder = forward_decoder
        self.backward_decoder = backward_decoder

        self.encoder_type = self.encoder.name
        self.decoder_type = self.forward_decoder.name
        assert self.forward_decoder.name == self.backward_decoder.name, " - Types of two decoders should be the same. "
    
    def forward(self, inputs):
        #=========== forward ===========#
        # encoding
        inputs['forward decoder']['enc_output'], inputs['forward decoder']['hidden'] = self.encoder(inputs['forward encoder'])
        # decoding
        f_dec_output = self.forward_decoder(inputs['forward decoder'], generator=self.forward_generator)
        f_dec_output['pred'] = self.forward_generator(f_dec_output['pred'])
        
        #=========== backward ===========#
        # encoding
        inputs['backward decoder']['enc_output'], inputs['backward decoder']['hidden'] = self.encoder(inputs['backward encoder'])
        # decoding
        b_dec_output = self.backward_decoder(inputs['backward decoder'], generator=self.backward_generator)
        b_dec_output['pred'] = self.backward_generator(b_dec_output['pred'])

        return f_dec_output, b_dec_output


class OpenNQG(nn.Module):
    '''
    A seq2seq-based Question Generation model 
    utilize structures of both RNN and Transformer

    Input: (1) src_seq: rnn_enc——src_seq,lengths; transf_enc——src_seq,src_pos 
           (2) tgt_seq: rnn_dec——tgt_seq; transf_dec——tgt_seq,tgt_pos
           (3) src_sep
           (4) feat_seqs: list of feat_seq
           (5) ans_seq: (ans_seq,ans_feat_seqs)/ans_seq,ans_lengths
           (6) max_length: max length of sentences in src (answer not included) ——> for DataParallel Class

    Output: results output from the Decoder (type: dict)
    '''
    def __init__(self, encoder, decoder, answer_encoder=None):
        super(OpenNQG, self).__init__()
        
        self.encoder = encoder
        self.decoder = decoder

        # self.answer = True if answer_encoder else False
        # if self.answer:
        #     self.answer_encoder = answer_encoder
        
        self.encoder_type = self.encoder.name
        self.decoder_type = self.decoder.name

    def forward(self, inputs, max_length=0, rl_type=''):
        #========== forward ==========#
        enc_output, hidden = self.encoder(inputs['encoder'])
        # if self.answer:
        #     _, hidden = self.answer_encoder(inputs['answer-encoder'])
        inputs['decoder']['enc_output'], inputs['decoder']['hidden'] = enc_output, hidden
        dec_output = self.decoder(inputs['decoder'], max_length=max_length, rl_type=rl_type, 
                                  generator=self.generator)
        #========== generate =========#
        dec_output['pred'] = self.generator(dec_output['pred'])

        return dec_output
