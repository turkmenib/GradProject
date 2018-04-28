# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 23:12:14 2018

@author: Ibo Turk
"""

from matplotlib import pyplot as plt

result_path = 'results/'

model, loss, acc, data_number, batch_size, epochs, test_set, info = "model", "loss", "acc", "data_number", "batch_size", "epochs", "test_set", "info"
tralos = 'tralos'
traacc = 'traacc'
numofnn = 0
with open("number.txt", 'r') as rptr:
    numofnn = int(rptr.readline())
with open(result_path + 'results.txt', "r") as rptr:
    DicArr = [dict() for k in range(numofnn)]
    for k in range(numofnn):
        DicArr[k][loss] = [0]
        DicArr[k][acc] = [0]
        DicArr[k][batch_size] = []
        DicArr[k][epochs] = [0]
        DicArr[k][tralos] = []
        DicArr[k][traacc] = []
    for line in rptr:
        print("original", line)
        line = line.replace('{', '')
        line = line.replace('[', '')
        line = line.replace(',', '')
        line = line.replace(']', '')
        line = line.replace('}', ' ')
        line = line.split()
        line.remove("'loss':")
        line.remove("'acc':")
        for x in range(len(line)):
            try:
                line[x] = int(line[x])
            except ValueError:
                try:
                    line[x] = float(line[x])
                except ValueError:
                    line[x] = line[x] +',' + line[x+1]+ ',' + line[x+2]
                    line.pop()
                    line.pop()
                    break
        index = line[-8] - 1
        enum = line[-3]
        DicArr[index][info] = line[-1]
        DicArr[index][test_set] = line[-2]
        DicArr[index][epochs].append(enum + max(DicArr[index][epochs]))
        DicArr[index][batch_size].append(line[-4])
        DicArr[index][data_number] = line[-5]
        DicArr[index][acc].append(line[-6])
        DicArr[index][loss].append(line[-7])
        DicArr[index][model] = line[-8]
        
        for e in range(enum):
            DicArr[index][tralos].append(line[e])
            DicArr[index][traacc].append(line[enum + e])
    
    for k in DicArr:
        print(k)
    
    for k in range(len(DicArr)):
        x1 = DicArr[k][epochs]
        ac = DicArr[k][acc]
        los = DicArr[k][loss]
        x2 = [i for i in range(1, max(x1)+1)]
        tacc = DicArr[k][traacc]
        tloss = DicArr[k][tralos]
        x1.remove(0)
        ac.remove(0)
        los.remove(0)
        plt.subplot(numofnn, 2, 2*k + 1), plt.title("Accuracy")
        plt.plot(x1, ac,'r--', x2, tacc)
        plt.subplot(numofnn, 2, 2*k + 2), plt.title("Loss")
        plt.plot(x1, los, 'r--', x2, tloss)
    plt.show()
        