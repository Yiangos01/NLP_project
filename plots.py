import sys
import numpy as np
import itertools
from collections import Counter
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
import matplotlib.pyplot as plt

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
  print(distance_sum,len(row))
  if len(row)!=1: 
   distance_sum=float(distance_sum)/len(row)-1
  else:
   distance_sum=float(distance_sum)/len(row)
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

def showAttention(input_sentence, output_words, attentions):
 # Set up figure with colorbar
 fig = plt.figure()
 ax = fig.add_subplot(111)
 cax = ax.matshow(attentions.numpy(), cmap='bone')
 fig.colorbar(cax)

 # Set up axes
 ax.set_xticklabels([''] + input_sentence.split(' ') +
        ['<EOS>'], rotation=90)
 ax.set_yticklabels([''] + output_words)

 # Show label at every tick
 ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
 ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

 plt.show()

def bleu_score(candidate,reference):
 # print(candidate)
 # print(reference)
 # candidate_list = candidate.split()
 candidate_list = candidate.lower().split()
 reference_list = reference.lower().split()
 candidate_counter = Counter(candidate_list)
 reference_counter = Counter(reference_list)
 # print(candidate_counter,reference_counter)
 blue = 0
 for word in reference_list:
     # print(word,min(candidate_counter[word], reference_counter[word]))
    blue += min(candidate_counter[word], reference_counter[word])
  # print(blue)
    blue = float(blue)/len( candidate_list)
 # print("last : ",len( candidate_list))
 return blue
    
def main():
    print (sys.argv)
    src_data=open("valid_src_final_big_preprocesed.txt")
    pred_data=open("big_pred.txt")
    tgt_data=open("valid_tgt_final_big_preprocesed.txt")
    dependencies=load_dependencies("dependencies_final.txt")
    ADD1,max_dist=calculate_ADD(dependencies)
    ADD=[]
    for i in ADD1:
        ADD.append(i - i % 0.05)
    print(len(ADD),len(max_dist))
    blue_score=[]
    sent_length=[]
    count=0
    Blue=[]
    ADD_30=[]
    sent_30=[]
    bleu_30=[]
    count=0
    for pred, tgt, src in itertools.izip(pred_data, tgt_data,src_data):
        pred=pred.lower()
        tgt=tgt.lower()
        sent_length.append(len(src.split()))
        bleu = sentence_bleu
        cc = SmoothingFunction()
        blue_score.append(bleu([tgt], pred, smoothing_function=cc.method4))#, smoothing_function=c.method4)) 
        if len(src.split()) >= 25 and len(src.split()) <= 25:
            print("in")
            sent_30.append(len(src.split()))
            ADD_30.append(ADD[count])
            bleu_30.append(bleu([tgt], pred, smoothing_function=cc.method4))
        count+=1

    f, subpl = plt.subplots(2, 2)
    # print(ADD_30)
    sent_length, bleu_se = zip(*sorted((xVal, np.mean([yVal for a, yVal in zip(sent_length, blue_score) if xVal==a])) for xVal in set(sent_length)))
    ADD, bleu_ad = zip(*sorted((xVal, np.mean([yVal for a, yVal in zip(ADD, blue_score) if xVal==a])) for xVal in set(ADD)))
    sent_30_1, bleu_30_1 = zip(*sorted((xVal, np.mean([yVal for a, yVal in zip(sent_30, bleu_30) if xVal==a])) for xVal in set(sent_30)))
    ADD_30, bleu_30 = zip(*sorted((xVal, np.mean([yVal for a, yVal in zip(ADD_30, bleu_30) if xVal==a])) for xVal in set(ADD_30)))
    subpl[0][0].plot(ADD[:-3], bleu_ad[:-3])
 # subpl[0, 0].set_title('Add & Pred score')
    subpl[0][0].set_xlabel('ADD')
    subpl[0][0].set_ylabel('blue_score')
    subpl[1][0].plot(sent_length[:52], bleu_se[:52])
 # subpl[0, 1].set_title('max_dist & Pred score')
    subpl[1][0].set_xlabel('sent_length')
    subpl[1][0].set_ylabel('blue_score')
    subpl[0][1].scatter(ADD_30,bleu_30)
 # subpl[0, 1].set_title('max_dist & Pred score')
    subpl[0][1].set_xlabel('25 lenght sentences ADD')
    subpl[0][1].set_ylabel('blue_score')
    subpl[1][1].scatter(sent_30_1, bleu_30_1)
 # subpl[0, 1].set_title('max_dist & Pred score')
    subpl[1][1].set_xlabel('30-35 lenght sentences')
    subpl[1][1].set_ylabel('blue_score')
 
    plt.show()


if __name__ == "__main__":
 main()


