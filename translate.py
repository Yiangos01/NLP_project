#!/usr/bin/env python

from __future__ import division
from builtins import bytes
import os
import argparse
import math
import codecs
import torch
import csv
import numpy as np
import onmt
import onmt.IO
import opts
from itertools import takewhile, count
import matplotlib.pyplot as plt
try:
    from itertools import zip_longest
except ImportError:
    from itertools import izip_longest as zip_longest

parser = argparse.ArgumentParser(
    description='translate.py',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
opts.add_md_help_argument(parser)
opts.translate_opts(parser)

opt = parser.parse_args()


def report_score(name, score_total, words_total):
    print("%s AVG SCORE: %.4f, %s PPL: %.4f" % (
        name, score_total / words_total,
        name, math.exp(-score_total/words_total)))


def get_src_words(src_indices, index2str):
    words = []
    raw_words = (index2str[i] for i in src_indices)
    words = takewhile(lambda w: w != onmt.IO.PAD_WORD, raw_words)
    return " ".join(words)

def calculate_ADD(input):
    scores=[]
    max_d=[]
    for i,row in enumerate(input):
        max_dist=-np.inf
        distance_sum=0
        for dependency in row[:-1]:
            
            d=dependency
            d=d.replace('\'', '')
            d=d.replace('"', '')
            d=d.split()
            pos1=d[1].split("-")
            pos2=d[2].split("-")
            distance=np.abs(int(pos1[len(pos1)-1])-int(pos2[len(pos2)-1]))
            if distance>max_dist:
                max_dist=distance
            distance_sum+=distance
        
        if len(row)!=1:    
            distance_sum/=len(row)-1
        else:
            distance_sum/=len(row)
        scores.append(distance_sum)
        max_d.append(max_dist)
    return scores,max_d
    
def load_dependencies(input):
    dependencies=[]
    with open(input, 'r') as csvfile:
       
        dependencies=[]
        for i,row in enumerate(csvfile):
            try:
                row = str(row).replace(']', '')
                row = str(row).replace('[', '')
                row = str(row).replace('(', ' ')
                row = str(row).replace(',', '')
                row = str(row).replace('\n', '')
                dependencies.append(str(row).split(")"))
            except:
                print("error dependencies")
                print(row)

    return dependencies

def main():
    file_name="dependencies_final.txt"
    dependencies=load_dependencies(file_name)
    ADD,max_dist=calculate_ADD(dependencies)
    print(len(ADD),len(max_dist))
   
    dummy_parser = argparse.ArgumentParser(description='train.py')
    opts.model_opts(dummy_parser)
    dummy_opt = dummy_parser.parse_known_args([])[0]

    opt.cuda = opt.gpu > -1
    if opt.cuda:
        torch.cuda.set_device(opt.gpu)
    translator = onmt.Translator(opt, dummy_opt.__dict__)
    out_file = codecs.open(opt.output, 'w', 'utf-8')
    pred_score_total, pred_words_total = 0, 0
    gold_score_total, gold_words_total = 0, 0
    if opt.dump_beam != "":
        import json
        translator.initBeamAccum()
    data = onmt.IO.ONMTDataset(
        opt.src, opt.tgt, translator.fields,
        use_filter_pred=False)

    test_data = onmt.IO.OrderedIterator(
        dataset=data, device=opt.gpu,
        batch_size=opt.batch_size, train=False, sort=False,
        shuffle=False)

    
    
    
    preds=[]
    gold_preds=[]
    counter = count(1)
    for batch in test_data:
        pred_batch, gold_batch, pred_scores, gold_scores, attn, src \
            = translator.translate(batch, data)
        pred_score_total += sum(score[0] for score in pred_scores)
        pred_words_total += sum(len(x[0]) for x in pred_batch)
        if opt.tgt:
            gold_score_total += sum(gold_scores)
            gold_words_total += sum(len(x) for x in batch.tgt[1:])

        # z_batch: an iterator over the predictions, their scores,
        # the gold sentence, its score, and the source sentence for each
        # sentence in the batch. It has to be zip_longest instead of
        # plain-old zip because the gold_batch has length 0 if the target
        # is not included.
        z_batch = zip_longest(
                pred_batch, gold_batch,
                pred_scores, gold_scores,
                (sent.squeeze(1) for sent in src.split(1, dim=1)))
        
        for pred_sents, gold_sent, pred_score, gold_score, src_sent in z_batch:
            n_best_preds = [" ".join(pred) for pred in pred_sents[:opt.n_best]]
            # print(get_src_words(src_sent, translator.fields["src"].vocab.itos))
            # print("sent_number ",src_sent)
            out_file.write('\n'.join(n_best_preds))
            out_file.write('\n')
            out_file.flush()

            if opt.verbose:
                sent_number = next(counter)
                words = get_src_words(src_sent, translator.fields["src"].vocab.itos)
                os.write(1, bytes('\nSENT %d: %s\n' % (sent_number, words), 'UTF-8'))
                
                best_pred = n_best_preds[0]
                best_score = pred_score[0]
                os.write(1, bytes('PRED %d: %s\n' % (sent_number, best_pred), 'UTF-8'))
                print("PRED SCORE: %.4f" % best_score)
                preds.append(best_score)
                
                
                if opt.tgt:
                    print("gold_sent: , "+str(gold_sent))
                    tgt_sent = ' '.join(gold_sent)
                    os.write(1, bytes('GOLD %d: %s\n' %
                             (sent_number, tgt_sent), 'UTF-8'))
                    print("GOLD SCORE: %.4f" % gold_score)
                    gold_preds.append(gold_score)
                if len(n_best_preds) > 1:
                    print('\nBEST HYP:')
                    for score, sent in zip(pred_score, n_best_preds):
                        os.write(1, bytes("[%.4f] %s\n" % (score, sent),
                                 'UTF-8'))

    f, subpl = plt.subplots(2, 2)
    subpl[0, 0].scatter(ADD, preds)
    # subpl[0, 0].set_title('Add & Pred score')
    subpl[0, 0].set_xlabel('ADD')
    subpl[0, 0].set_ylabel('Pred score')
    subpl[0, 1].scatter(max_dist, preds)
    # subpl[0, 1].set_title('max_dist & Pred score')
    subpl[0, 1].set_xlabel('max_dist')
    subpl[0, 1].set_ylabel('Pred score')
    subpl[1, 0].scatter(ADD,gold_preds)
    # subpl[1, 0].set_title('Add & Gold')
    subpl[1, 0].set_xlabel('ADD')
    subpl[1, 0].set_ylabel('Gold score')
    subpl[1, 1].scatter(max_dist,gold_preds)
    # subpl[1, 1].set_title('max_dist & Gold')
    subpl[1, 1].set_xlabel('max_dist')
    subpl[1, 1].set_ylabel('Gold score')
    plt.show()

    report_score('PRED', pred_score_total, pred_words_total)
    if opt.tgt:
        report_score('GOLD', gold_score_total, gold_words_total)

    if opt.dump_beam:
        json.dump(translator.beam_accum,
                  codecs.open(opt.dump_beam, 'w', 'utf-8'))


if __name__ == "__main__":
    main()
