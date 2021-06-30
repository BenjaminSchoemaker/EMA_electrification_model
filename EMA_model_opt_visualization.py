# -*- coding: utf-8 -*-
"""
Created on Sun May  9 12:13:10 2021

@author: benjaminschoemaker
"""
import pandas as pd
import numpy as np 
from ema_workbench.analysis import parcoords
import seaborn as sns
import matplotlib.pyplot as plt
#Plot parallel coordinates
    
def reorder_data(df_orgnl,metric=None,DPR=None):
    df = df_orgnl.copy()
    if DPR == 1:
        df = df.drop(columns=['EHS grid connection costs',
                          'Variable transport tariff per kW per month furnace and compressor',
                          'Variable transport tariff per kW per month boiler']) 
    else:
        df = df.drop(columns=['DPR switch',
                              'EHS grid connection costs',
                              'Variable transport tariff per kW per month furnace and compressor',
                              'Variable transport tariff per kW per month boiler']) 
    if metric == 1:
        df_new = df.rename(columns={'"Base fee SDE++ per kWh"':'SDE',
                                      'Carbon levy increase rate':'CO2 levy', 
                                      'Energy tax electricity':'Electricity tax',
                                      'Carbon levy increase rate after 2030':'CO2 levy t>2030', 
                                      'Max. cum. emissions':'Max. emissions',
                                      'Max. cum. production costs':'Max. prod. costs',
                                      'Max. cum. policy costs':'Max. pol. costs'})
        df_new['Max. emissions'] = df_new['Max. emissions']/10e6
        df_new['Max. prod. costs'] = df_new['Max. prod. costs']/10e9
        df_new['Max. pol. costs'] = df_new['Max. pol. costs']/10e9
        
        df_new = df_new.sort_values(by=['Max. emissions'])
    
    elif metric==2:
        df_new = df.rename(columns={'"Base fee SDE++ per kWh"':'SDE',
                                      'Carbon levy increase rate':'CO2 levy', 
                                      'Energy tax electricity':'Electricity tax',
                                      'Carbon levy increase rate after 2030':'CO2 levy t>2030', 
                                      'Mean cum. emissions':'Mean emissions',
                                      'Dev. cum. emissions':'Dev. emissions',
                                      'Mean cum. production costs':'Mean prod. costs',
                                      'Dev. cum. production costs':'Dev. prod. costs',
                                      'Mean cum. policy costs':'Mean pol. costs',
                                      'Dev. cum. policy costs':'Dev. pol. costs',})
        df_new['Mean emissions'] = df_new['Mean emissions']/10e6
        df_new['Mean prod. costs'] = df_new['Mean prod. costs']/10e9
        df_new['Mean pol. costs'] = df_new['Mean pol. costs']/10e9
        df_new['Dev. emissions'] = df_new['Dev. emissions']/10e12
        df_new['Dev. prod. costs'] = df_new['Dev. prod. costs']/10e21
        df_new['Dev. pol. costs'] = df_new['Dev. pol. costs']/10e18
        
        df_new = df_new.sort_values(by=['Mean emissions'])
        
    elif metric==3:
        df_new = df.rename(columns={'"Base fee SDE++ per kWh"':'SDE',
                                      'Carbon levy increase rate':'CO2 levy', 
                                      'Energy tax electricity':'Electricity tax',
                                      'Carbon levy increase rate after 2030':'CO2 levy t>2030', 
                                      'Mean cum. emissions':'Mean emissions',
                                      'Peakedness cum. emissions':'P emissions',
                                      'Mean cum. production costs':'Mean prod. costs',
                                      'Peakedness cum. production costs':'P prod. costs',
                                      'Mean cum. policy costs':'Mean pol. costs',
                                      'Peakedness cum. policy costs':'P pol. costs',})
        
        df_new['Mean emissions'] = df_new['Mean emissions']/10e6
        df_new['Mean prod. costs'] = df_new['Mean prod. costs']/10e9
        df_new['Mean pol. costs'] = df_new['Mean pol. costs']/10e9
        
        df_new = df_new.sort_values(by=['Mean emissions'])
        
    return df_new
    
    
    # for i in range (0,len(df_new)):
    #     index = df_new.columns.get_loc('DPR switch')
    #     if df_new.iloc[i,index] == "Category('0', 0)":
    #         df_new.iloc[i,index] = 0.0
    #     else:
    #         df_new.iloc[i,index] = 1.0

# counter = 0 
# index = results1.columns.get_loc('DPR switch')
# for i in range (0,len(results1)):
#     if results1.iloc[i,index] == "Category('0', 0)":
#         counter += 1


results1 = pd.read_csv('output_data\\robust_results1.csv').drop(columns='Unnamed: 0') 
#results2 = pd.read_csv('output_data\\robust_results2.csv').drop(columns='Unnamed: 0') 
#results3 = pd.read_csv('output_data\\robust_results3.csv').drop(columns='Unnamed: 0') 
        
results1_1 = reorder_data(results1,metric=1)
results1_1_01 = results1_1.iloc[227:237,:]
results1_1_0 = results1_1.iloc[206:227,:]
results1_1_1 = results1_1.iloc[185:206,:]
results1_1_2 = results1_1.iloc[164:185,:]
results1_1_3 = results1_1.iloc[143:164,:]
results1_1_4 = results1_1.iloc[122:143,:]
results1_1_5 = results1_1.iloc[101:122,:]
results1_1_6 = results1_1.iloc[80:101,:]
results1_1_7 = results1_1.iloc[59:80,:]
results1_1_8 = results1_1.iloc[38:59,:]
results1_1_9 = results1_1.iloc[17:38,:]
results1_1_10 = results1_1.iloc[0:17,:]

# results1_DPR = reorder_data(results1,metric=1,DPR=1)
# results1_DPR_on = results1_DPR.loc[(results1_DPR['DPR switch'] == "Category('1', 1)")]
# results1_DPR_on_df = results1_DPR_on.drop(columns='DPR switch') 

#paraxes = parcoords.ParallelAxes(parcoords.get_limits(results1_DPR_on_df), rot=90)

# results1_1_1 = results1_1.loc[(results1_1['Max. emissions'] > 4.75)]
# results1_1_2 = results1_1.loc[(results1_1['Max. emissions'] < 4.75) & (results1_1['Max. emissions'] > 4.70)]
# results1_1_3 = results1_1.loc[(results1_1['Max. emissions'] < 4.70) & (results1_1['Max. emissions'] > 4.65)]
# results1_1_4 = results1_1.loc[(results1_1['Max. emissions'] < 4.65) & (results1_1['Max. emissions'] > 4.60)]
# results1_1_5 = results1_1.loc[(results1_1['Max. emissions'] < 4.60) & (results1_1['Max. emissions'] > 4.55)]
# results1_1_6 = results1_1.loc[(results1_1['Max. emissions'] < 4.55) & (results1_1['Max. emissions'] > 4.50)]
# results1_1_7 = results1_1.loc[(results1_1['Max. emissions'] < 4.50)]

paraxes = parcoords.ParallelAxes(parcoords.get_limits(results1_1), rot=90)
results1_1_minmaxCO2 = results1_1.loc[(results1_1['Max. emissions'] == np.min(results1_1['Max. emissions']))]
results1_1_minmaxProd = results1_1.loc[(results1_1['Max. prod. costs'] == np.min(results1_1['Max. prod. costs']))]
results1_1_minmaxPol = results1_1.loc[(results1_1['Max. pol. costs'] == np.min(results1_1['Max. pol. costs']))]

paraxes = parcoords.ParallelAxes(parcoords.get_limits(results1_1), rot=90)
paraxes.plot(results1_1, color=sns.color_palette()[0])
paraxes.plot(results1_1_minmaxCO2, color=sns.color_palette()[1], label='CO2')
paraxes.plot(results1_1_minmaxProd, color=sns.color_palette()[6], label='Prod.')
paraxes.plot(results1_1_minmaxPol, color=sns.color_palette()[3], label='Pol.')
paraxes.legend()

# paraxes = parcoords.ParallelAxes(parcoords.get_limits(results1_1), rot=90)
# paraxes.plot(results1_1, color=sns.color_palette()[0])
# paraxes.plot(results1_DPR_on_df, color=sns.color_palette()[1])
paraxes = parcoords.ParallelAxes(parcoords.get_limits(results1_1), rot=90)
paraxes.plot(results1_1, color=sns.color_palette()[0])
paraxes.plot(results1_1_01, color=sns.color_palette()[1])
plt.savefig('parcoords1_1',dpi=300, bbox_inches='tight')

paraxes = parcoords.ParallelAxes(parcoords.get_limits(results1_1), rot=90)
paraxes.plot(results1_1, color=sns.color_palette()[0])
paraxes.plot(results1_1_0, color=sns.color_palette()[1])

paraxes = parcoords.ParallelAxes(parcoords.get_limits(results1_1), rot=90)
paraxes.plot(results1_1, color=sns.color_palette()[0])
paraxes.plot(results1_1_1, color=sns.color_palette()[1])

paraxes = parcoords.ParallelAxes(parcoords.get_limits(results1_1), rot=90)
paraxes.plot(results1_1, color=sns.color_palette()[0])
paraxes.plot(results1_1_2, color=sns.color_palette()[3], label='>80p')

paraxes = parcoords.ParallelAxes(parcoords.get_limits(results1_1), rot=90)
paraxes.plot(results1_1, color=sns.color_palette()[0])
paraxes.plot(results1_1_3, color=sns.color_palette()[6], label='>70p')

paraxes = parcoords.ParallelAxes(parcoords.get_limits(results1_1), rot=90)
paraxes.plot(results1_1, color=sns.color_palette()[0])
paraxes.plot(results1_1_4, color=sns.color_palette()[1], label='>60p')

paraxes = parcoords.ParallelAxes(parcoords.get_limits(results1_1), rot=90)
paraxes.plot(results1_1, color=sns.color_palette()[0])
paraxes.plot(results1_1_5, color=sns.color_palette()[3], label='>50p')
# results1_1_5_minimaxP = results1_1_5.loc[(results1_1_5['Max. prod. costs'] == np.min(results1_1_5['Max. prod. costs']))]
# paraxes.plot(results1_1_5_minimaxP, color=sns.color_palette()[0])

paraxes = parcoords.ParallelAxes(parcoords.get_limits(results1_1), rot=90)
paraxes.plot(results1_1, color=sns.color_palette()[0])
paraxes.plot(results1_1_6, color=sns.color_palette()[6], label='>40p')

paraxes = parcoords.ParallelAxes(parcoords.get_limits(results1_1), rot=90)
paraxes.plot(results1_1, color=sns.color_palette()[0])
paraxes.plot(results1_1_7, color=sns.color_palette()[1], label='>30p')

paraxes = parcoords.ParallelAxes(parcoords.get_limits(results1_1), rot=90)
paraxes.plot(results1_1, color=sns.color_palette()[0])
paraxes.plot(results1_1_8, color=sns.color_palette()[3], label='>20p')

paraxes = parcoords.ParallelAxes(parcoords.get_limits(results1_1), rot=90)
paraxes.plot(results1_1, color=sns.color_palette()[0])
paraxes.plot(results1_1_9, color=sns.color_palette()[6], label='>10p')

paraxes = parcoords.ParallelAxes(parcoords.get_limits(results1_1), rot=90)
paraxes.plot(results1_1, color=sns.color_palette()[0])
paraxes.plot(results1_1_10, color=sns.color_palette()[1], label='<10p')


paraxes.legend()
     
        
    # paraxes = parcoords.ParallelAxes(parcoords.get_limits(results_1), rot=45)
    # paraxes.plot(results_1)