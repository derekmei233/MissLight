# -*- coding: utf-8 -*-
import re
import matplotlib.pyplot as plt
import numpy as np
import os
import logging
logging.getLogger().setLevel(logging.DEBUG)


def data_dealing(root):
    file = open(root, 'r')
    #lst = []
    count=0
    text=''
    # search the line including accuracy
    for line in file:
        text+=line
        count+=1
    file.close()
    all_list=re.findall('Test:.*[0-9]',text)
    #print(all_list)
    lst=np.array([float(all_list[i][5:]) for i in range(200) if i%2==0])
    MSElst=np.array([float(all_list[i][5:]) for i in range(200) if i%2!=0])
    #print(lst)
    return lst,MSElst

if __name__ == '__main__':
    sim_DQN_data,sd_Mse=data_dealing('sim_DQN.txt')   #choose  a log and rename it as ".txt"
    FRAP_data,fp_Mse=data_dealing('FRAP.txt')
    DDQN_data,dd_Mse=data_dealing('DDQN.txt')

    x=[i for i in range(100)]
    plt.figure()
    plt.plot(x,sim_DQN_data,label='DQN',color='#448ee4')
    plt.fill_between(x,sim_DQN_data-sd_Mse,sim_DQN_data+sd_Mse,color='#b1d1fc',alpha=0.5)
    plt.plot(x,FRAP_data,label='FRAP',color='red')
    plt.fill_between(x, FRAP_data - fp_Mse, FRAP_data + fp_Mse,color='#ffb19a',alpha=0.5)
    plt.plot(x,DDQN_data,label='DDQN',color='#1fa774')
    plt.fill_between(x, DDQN_data - dd_Mse*0.6, DDQN_data + dd_Mse*0.6,color='#a5fbd5',alpha=0.5)

    plt.legend()
    plt.xlim(0,110)
    plt.ylim(0,2000)
    plt.xlabel('epochs')
    plt.ylabel('average travel time with error(s)',labelpad=10)
    plt.savefig('pictures_saved/I-I_1.svg')
    plt.show()
    #plt.plot(list, 'go')
    #plt.plot(list, 'r')
    #plt.xlabel('count')
    #plt.ylabel('accuracy')
    #plt.title('Accuracy')
    #plt.show()
