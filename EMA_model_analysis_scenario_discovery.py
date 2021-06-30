# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 14:44:12 2021

@author: benjaminschoemaker
"""

# more or less default imports when using
# the workbench
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

from ema_workbench import (ema_logging,load_results)

from ema_workbench.analysis import prim

ema_logging.log_to_stderr(ema_logging.INFO)

wd = os.getcwd()

results = load_results('output_data\\results_300scenarios_100policies.tar.gz')

def save_fig(fig, dir, name, dpi=300):
    fig.savefig('{}/fig_{}_{}dpi.png'.format(dir, name, dpi), dpi=dpi,
                bbox_inches='tight', format='png')
    
experiments, outcomes = results

def perform_prim(exp,out,column,p_of_interest,title,box_no):
    x = exp.drop(columns='policy')
    p_frac = p_of_interest/100
    interest_threshold = pd.DataFrame(out[column]).quantile(p_frac)
    y = out[column] < interest_threshold[0]
    prim_alg = prim.Prim(x, y, threshold=0.8)
    box = prim_alg.find_box()
    prim_tradeoff = box.show_tradeoff()
    plt.show()
    save_fig(prim_tradeoff,wd+'\\plots\\scenario_discovery','prim_tradeoff_'+title)
    
    box.inspect(box_no)
    prim_box = box.inspect(box_no, style='graph')
    save_fig(prim_box,wd+'\\plots\\scenario_discovery','prim_box_'+title)
    plt.show()
    
    # prim_scatter = box.show_pairs_scatter(box_no)
    # save_fig(prim_scatter,wd+'\\plots\\scenario_discovery','prim_scatter_'+title)
    # plt.show()
    
    # box2 = prim_alg.find_box()
    # prim_tradeoff2 = box2.show_tradeoff()
    # plt.show()
    # save_fig(prim_tradeoff2,wd+'\\plots\\scenario_discovery','prim_tradeoff2_'+title)
    
    # box2.inspect(box_no)
    # prim_box2 = box2.inspect(box_no, style='graph')
    # save_fig(prim_box2,wd+'\\plots\\scenario_discovery','prim_box2_'+title)
    # plt.show()
    

perform_prim(experiments,outcomes,'Cum. emissions 2050',20,'emissions',26)
perform_prim(experiments,outcomes,'Cum. production costs 2050',20,'production_costs',26)
perform_prim(experiments,outcomes,'Cum. policy costs 2050',40,'policy_costs',16)

def perform_prim_worst_cases(exp,out,column,p_of_interest,title,box_no):
    x = exp.drop(columns='policy')
    p_frac = p_of_interest/100
    interest_threshold = pd.DataFrame(out[column]).quantile(p_frac)
    y = out[column] > interest_threshold[0]
    prim_alg = prim.Prim(x, y, threshold=0.8)
    box = prim_alg.find_box()
    prim_tradeoff = box.show_tradeoff()
    plt.show()
    save_fig(prim_tradeoff,wd+'\\plots\\scenario_discovery','prim_tradeoff_worst'+title)
    
    box.inspect(box_no)
    prim_box = box.inspect(box_no, style='graph')
    save_fig(prim_box,wd+'\\plots\\scenario_discovery','prim_box_worst'+title)
    plt.show()

perform_prim_worst_cases(experiments,outcomes,'Cum. emissions 2050',80,'emissions',26)
# perform_prim_worst_cases(experiments,outcomes,'Cum. production costs 2050',80,'production_costs',26)
# perform_prim_worst_cases(experiments,outcomes,'Cum. policy costs 2050',60,'policy_costs',16)

def perform_prim_mid_cases(exp,out,column,p_upper,p_lower,title,box_no):
    x = exp.drop(columns='policy')
    p_frac_upper = p_upper/100
    p_frac_lower = p_lower/100
    high_limit = pd.DataFrame(out[column]).quantile(p_frac_upper)
    low_limit = pd.DataFrame(out[column]).quantile(p_frac_lower)
    y_low = out[column] < high_limit[0]
    y_high = out[column] > low_limit[0]
    y = np.ones(len(out[column]), dtype=bool)
    for i in range(0,len(out[column])):
        if y_low[i] == True and y_high[i] == True:
            y[i] == True
        else:
            y[i] = False
            
    prim_alg = prim.Prim(x, y, threshold=0.8)
    box = prim_alg.find_box()
    prim_tradeoff = box.show_tradeoff()
    plt.show()
    save_fig(prim_tradeoff,wd+'\\plots\\scenario_discovery','prim_tradeoff_mid'+title)
    
    box.inspect(box_no)
    prim_box = box.inspect(box_no, style='graph')
    save_fig(prim_box,wd+'\\plots\\scenario_discovery','prim_box_mid'+title)
    plt.show()

perform_prim_mid_cases(experiments,outcomes,'Cum. emissions 2050',80,50,'emissions',26)


# #Emissions
# x = experiments.drop(columns='policy')
# p15_em = pd.DataFrame(outcomes['Cum. emissions 2050']).quantile(0.15)
# y_em = outcomes['Cum. emissions 2050'] < p15_em[0]
# prim_alg_emissions = prim.Prim(x, y_em, threshold=0.8)
# box_emissions = prim_alg_emissions.find_box()
# prim_tradeoff_emissions = box_emissions.show_tradeoff()
# plt.show()
# save_fig(prim_tradeoff_emissions,wd,'prim_tradeoff_emissions')

# box_emissions.inspect(35)
# prim_box_emissions = box_emissions.inspect(35, style='graph')
# save_fig(prim_box_emissions,wd,'prim_box_emissions')
# plt.show()

# prim_scatter_emissions = box_emissions.show_pairs_scatter(35)
# save_fig(prim_scatter_emissions,wd,'prim_scatter_emissions')
# plt.show()
