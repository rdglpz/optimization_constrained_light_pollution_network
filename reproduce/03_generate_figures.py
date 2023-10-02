#Cargando librerías 
import importlib

import matplotlib.pyplot as plt

#from matplotlib import cm
#import matplotlib as mpl
#import itertools as it


#from matplotlib.colors import ListedColormap, LinearSegmentedColormap

import scipy as sp
import numpy as np
import pandas as pd
import os
import sys  

root = os.getcwd() + "/.."
sys.path.insert(0, root)

import src.positioning_sensors as ps
import src.regionGrowing as rg
import src.manageExperiments as me
import src.network_fitness as netfit

importlib.reload(ps)
importlib.reload(rg)
importlib.reload(me)
importlib.reload(netfit)

#from geneticalgorithm import geneticalgorithm as ga



import config.config as cfg
importlib.reload(cfg)


fn = "reproduce_paper_experiment.txt"
setup = me.readConfigFile(fn)


#Loading precalculated local and directed empirical variograms

filesv = (cfg.experiment + setup["experiment_id"] 
          + setup["folder_semivariances"] 
          + setup["output_semivariances"] 
          + ".csv"
         )

filesvmask = (cfg.experiment + setup["experiment_id"] 
              + setup["folder_semivariances"] 
              + setup["output_semivariances"] 
              + "mask.csv"
             )

OptimumValues = (cfg.experiment + setup["experiment_id"] 
                 + setup["folder_output"] 
                 + setup["output_values"]
                )

arguments = (cfg.experiment + setup["experiment_id"] 
             + setup["folder_output"] 
             + setup["output_args"]
            )

path2optimum_values_file = (cfg.experiment 
                           + setup["experiment_id"] 
                           + setup["folder_output"] 
                           )


data = pd.read_csv(filesv)
data_m = pd.read_csv(filesvmask)
variogram_set = np.array(data.iloc[:,3:])
variogram_set_m = np.array(data_m.iloc[:,3:])

#cargamos imagenes en luminance e importance
ilumina = setup["folder_input"] + setup["input_ntli"]
niveles = setup["folder_input"] + setup["input_evm"]
sigma_y = setup['filterg_sy']
sigma_x = setup['filterg_sx']
sigma = [sigma_y, sigma_x]


#NLTI: luminance
luminance = ps.readIMG(ilumina)

#EAM: Environtmental attention map is the importance
EAM = ps.readIMG(niveles, invert = True)

DNTLI, b = ps.desaturate(luminance, th = 63)

variograms = variogram_set.reshape(len(variogram_set), 
                                   DNTLI.shape[0], 
                                   DNTLI.shape[1]
                                  )
variograms_m = variogram_set_m.reshape(len(variogram_set), 
                                       DNTLI.shape[0],DNTLI.shape[1])

coords = np.array(data.iloc[:, 1:3])

FDNTLI = sp.ndimage.gaussian_filter(DNTLI, sigma, 
                                    mode = setup['gaussian_mode']
                                   )

FDNTLI = (FDNTLI >= setup['neglect_values'])*FDNTLI

sensitivity = ps.f5(FDNTLI, EAM, 64)

#paths to ntli and vulnerability
ilumina = setup["folder_input"] + setup["input_ntli"]
niveles = setup["folder_input"] + setup["input_evm"]
sigma_y = setup['filterg_sy']
sigma_x = setup['filterg_sx']
sigma = [sigma_y,sigma_x]

aptitude = netfit.NetworkFitness(FDNTLI, EAM, sensitivity, variograms, 
                                 variograms_m, coords)
aptitude.selectFitnessFunction("max")


res_read = pd.read_csv(OptimumValues)
rr = np.array(res_read)
r = list([])
for i,j in enumerate(rr):
    r.append(j[1:(1)+((i % 7) +1)*2])

R = (aptitude.project(r[48])+(FDNTLI>0)*1).astype(int)

def plotSolutions(p,img):

    z = np.copy(img)
    cs = int(9)

    a = p.reshape(-1,2)
    for b in a:
        c = b.astype(int)
        z[tuple(c)] = 2
        z[tuple([c[0]+1,c[1]])]=cs
        z[tuple([c[0]+1,c[1]+1])]=cs
        z[tuple([c[0],c[1]+1])]=cs
        z[tuple([c[0]-1,c[1]-1])]=cs
        z[tuple([c[0]-1,c[1]])]=cs
        z[tuple([c[0]-1,c[1]+1])]=cs
        z[tuple([c[0],c[1]-1])]=cs
        z[tuple([c[0]+1,c[1]-1])]=cs
        

    return z

P = plotSolutions(r[2],R)


R = plotSolutions(r[48], (FDNTLI>0)*1).astype(int)   


fig, axs = plt.subplots(ncols=7, nrows=7, figsize=(15, 15), 
                        constrained_layout = True)


ix = 0
saveR = np.zeros(R.shape)
f = 0
c = setup["sensitivity_c"]
for row in range(len(c)):
    for col in range(7):
        R = (aptitude.project(r[ix])+(FDNTLI>0)*1).astype(int)
        PS = plotSolutions(r[ix],R)
        axs[row,col].imshow(PS,cmap="viridis",interpolation='none')

        if col==0: 
            axs[row, col].set_ylabel("$c={c: 1.0f}$".format(c = c[row]), 
                                     rotation = 0,fontsize = 18)
            axs[row, col].yaxis.set_label_coords(-.27, .5)
        if row==0: 
            axs[row, col].set_title('$n = {n: 1.0f}$'.format(n = col + 1), 
                                    fontsize = 18
                                    )
            
        axs[row, col].tick_params(left = False, right = False , 
                                  labelleft = False , labelbottom = False, 
                                  bottom = False
                                  )

        ix += 1


#save in results folder
print("Saving Figure 10 as:", path2optimum_values_file + "figure_10.pdf")
plt.savefig(path2optimum_values_file + "figure_10.pdf", format = 'pdf', 
            dpi = 500 , 
            bbox_inches = "tight")

aptitudes = np.array(pd.read_csv(arguments))[:, 1]

fig, ax = plt.subplots()
styles = ['--bo', '--', ':', '-.', '.-k', 'b', '-vm']
for i in range(7):
    
    ax.plot(-aptitudes.reshape(7, 7).T[:, i]*100, styles[i], 
            label = "c = " + str(c[i]))

ax.set_xticks([i for i in range(len(c))])
ax.set_xticklabels([i + 1 for i in range(len(c))])
ax.set_ylabel(r"Captured Sensitivity Percentage")
ax.set_xlabel("Number of sensors")
ax.legend()
print("Saving Figure 09 as:", path2optimum_values_file + "figure_09.pdf")
plt.savefig(path2optimum_values_file + "figure_09.pdf", format = 'pdf', 
            bbox_inches = "tight")

nonsat,b = ps.desaturate(luminance,th=setup["desaturation_th"])

variograms = variogram_set.reshape(len(variogram_set),nonsat.shape[0],
                                   nonsat.shape[1]
                                   )
variograms_m = variogram_set_m.reshape(len(variogram_set),nonsat.shape[0],
                                       nonsat.shape[1])
coords = np.array(data.iloc[:,1:3])
NLTI = sp.ndimage.gaussian_filter(nonsat, sigma, mode='constant')
NLTI = (NLTI>=setup["neglect_values"])*NLTI


sensitivity = ps.f5(NLTI,EAM,1)
nf = netfit.NetworkFitness(NLTI,EAM,sensitivity,variograms,variograms_m,coords)
nf.selectFitnessFunction("max")
f = nf.f

solutions = list([])

for rx in r:  
    solutions.append(f(rx))
    
M = np.array(solutions).reshape(7,7)

cf = 1 
fig, ax = plt.subplots()
styles = ['--bo', '--', ':', '-.', '.-k', 'b', '-vm']
for i in range(7):
    ax.plot(-M[:, i]*100, styles[i], label = "n = " + str(i + 1))

ax.set_xticks([i for i in range(len(c))])
ax.set_xticklabels(c)
ax.set_ylabel(r"Captured FNTLI Percentage")
ax.set_xlabel("$c$")
ax.legend(ncol = 2 ,bbox_to_anchor = (0.55, 0.4), loc = 'upper left', 
          borderaxespad = 0
          )
print("Saving Figure 11 as:", path2optimum_values_file + "figure_11.pdf")
plt.savefig(path2optimum_values_file + "Figure_11.pdf", format = 'pdf', 
            bbox_inches = "tight")


