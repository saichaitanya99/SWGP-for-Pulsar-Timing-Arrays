import corner
import numpy as np
import matplotlib.pyplot as plt

psrname="J1022+1001"

chain1 = np.loadtxt('./chain_J0034-0534_finaldatarun_wodmgp/chain_1.txt')
#chain2 = np.loadtxt('all_previous_runs/lofarstationwise_runs/chain_J1022+1001_allstations_withdm_allnoise/chain_1.txt')


with open('./chain_J0034-0534_finaldatarun_wodmgp/pars.txt') as f1:
    xs1 = f1.readlines()
    
'''    
with open('all_previous_runs/chain_J1022+1001_lofar_allnoise/pars.txt') as f2:
    xs2 = f2.readlines()
'''

xs11 = [i.split('\n',1) for i in xs1]
pars1 = ([xs11[i][0] for i in range(len(xs11))])
#print(pars1,'\n')
'''
xs22 = [i.split('\n',1) for i in xs2]
pars2 = ([xs22[i][0] for i in range(len(xs22))])
print(pars2,'\n')
'''

idxs1 = np.arange(len(pars1))
#j=0
pars_new1=[]
for j in range(len(idxs1)):
    pars_new1.append(pars1[idxs1[j]])
print(pars_new1)

'''
idxs2 = np.arange(len(pars2)-7,len(pars2))
#j=0
pars_new2=[]
for j in range(len(idxs2)):
    pars_new2.append(pars2[idxs2[j]])
print(pars_new2)
'''

burn1 = int(0.25 * chain1.shape[0])
#burn2 = int(0.25 * chain2.shape[0])
fig = corner.corner(chain1[burn1:, idxs1],bins =30, labels=[i.replace(psrname+"_", "") for i in pars_new1],hist_kwargs={'density':True}, plot_datapoints=True, color='r',show_titles=True)
#corner.corner(chain2[burn2:,idxs2], hist_kwargs={'density':True}, plot_datapoints=False, color='b', fig=fig, show_titles=True)
plt.show()
plt.savefig("corner_data_J0034+0534_wodmgp.png")
    
   


