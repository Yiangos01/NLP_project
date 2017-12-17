import numpy as np
from os import listdir
from os.path import isfile, join

def output_generator(dep,src, dependancy_for_classifying = 'nsubj'):

    src=open(src,'r')
    ouputs=[]
    for i,row in enumerate(src):
        distance_sum=0
        output=np.zeros(len(row.split()))
        # print(i,row.split())
        # print(dep[i])
        for dependency in dep[i]:
            d=dependency
            d=d.replace('\'', '')
            d=d.replace('"', '')
            d=d.split()
            # print(d[0])
            if dependancy_for_classifying in d[0]:
                # print(i,'   : ',d[0])
                pos1=d[1].split("-")
                pos2=d[2].split("-")
                # print(pos1,' | ',pos2)
                output[int(pos1[len(pos1)-1]) -1]=-1
                output[int(pos2[len(pos2)-1]) -1]=1
        ouputs.append(output)
    if np.sum(output) != 0 :
        print(row)
        print(dep[i])
        print(output)

        
    return ouputs

def load_dependencies(input):
    dependencies=[]
    with open(input, 'r') as csvfile:
       
        dependencies=[]
        for row in csvfile:
            try:
        # resultwords  = [word for char in row if word.lower() not in stopwords]
                row = str(row).replace(']', '')
                row = str(row).replace('[', '')
                row = str(row).replace('(', ' ')
                row = str(row).replace(',', '')
                row = str(row).replace('\n', '')
                dependencies.append(str(row).split(")")[:-1])
                
            except:
                print("error dependencies")
                print(row)

    return dependencies

    # [root(ROOT-0, thank-1), nsubj(chris-5, you-2), advmod(much-4, so-3), advmod(chris-5, much-4), xcomp(thank-1, chris-5), cc(chris-5, and-7), 
    # nsubj(honor-13, it-8), cop(honor-13, s-9), advmod(honor-13, truly-10), det(honor-13, a-11), amod(honor-13, great-12), xcomp(thank-1, honor-13),
    #  conj:and(chris-5, honor-13), mark(have-15, to-14), acl(honor-13, have-15), det(opportunity-17, the-16), dobj(have-15, opportunity-17),
    #   mark(come-19, to-18), acl(opportunity-17, come-19), case(stage-22, to-20), det(stage-22, this-21), nmod:to(come-19, stage-22), 
    # advmod(m-25, twice-23), compound(m-25, i-24), nummod(stage-22, m-25), advmod(grateful-27, extremely-26), amod(stage-22, grateful-27)]

dependencies = load_dependencies('new_data/test_nsubs_dependencies.txt')

# # # final = []
# # # a=0
# # # for dd in dependencies:
# # #     for d in dd:
# # #         s = d.split()
# # #         for z in s:
# # #             print(z)
# # #         # print (d,'\n')
# # #         a+=2
# # #         if a >10:
# # #             exit(0)

outputs=output_generator(dependencies,"new_data/test_src_preprocesed.txt",'conj')



outputs = np.concatenate(outputs)

# # outputs.reshape((1, 340198))

print('targets_size :',outputs.shape)

np.savetxt('test_conj_targets.txt', outputs)

# with open('valid_src_yiang.txt','rb') as f:
#     word_counter = 0
# for line in f:
#     word_counter = 0
#     for word in line.split():



# inputs = []

# onlyfiles = [f for f in listdir('encoder_outputs') if isfile(join('encoder_outputs', f))]
# count = 0
# for i, filename in enumerate(onlyfiles):
#     temp = np.loadtxt('encoder_outputs/'+filename)
#     count += temp.shape[0]
#     print(temp.shape)
#     inputs.append( temp)
#     # if i == 3:
#         # break

# print('count =' ,count)

# inputs = np.concatenate(inputs)

# print('inputs_size:',inputs.shape)



# np.savetxt('inputs.txt', inputs)
