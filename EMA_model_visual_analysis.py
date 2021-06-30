# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 16:41:27 2021

@author: benjaminschoemaker
"""

# more or less default imports when using
# the workbench
import seaborn as sns
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np

from ema_workbench import load_results

from ema_workbench.analysis import Density
from ema_workbench.analysis.plotting import lines


def value_end(timeseries):
    return timeseries[-1]

wd = os.getcwd()

results = load_results('output_data\\results_300scenarios_100policies_new.tar.gz')

experiments, outcomes = results


#Functions save_fig, change_fontsize and plot_outcomes are adapted from master thesis Erika van der Linden (2020): https://repository.tudelft.nl/islandora/object/uuid%3Ae51dbb87-09f7-4c33-a956-226874a1e7b7

def save_fig(fig, dir, name, dpi=300):
    fig.savefig('{}/fig_{}_{}dpi.png'.format(dir, name, dpi), dpi=dpi,
                bbox_inches='tight', format='png')
    
def change_fontsize(fig, fs=11.5):
    for ax in fig.axes:
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                      ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(fs)
        try:
            parasites = ax.parasites
        except AttributeError:
            pass
        else:
            for parisite in parasites:
                for axis in parisite.axis.values():
                    axis.major_ticklabels.set_fontsize(fs)
                    axis.label.set_fontsize(fs)
            for axis in ax.axis.values():
                axis.major_ticklabels.set_fontsize(fs)
                axis.label.set_fontsize(fs)
        if ax.legend_ != None:
            for entry in ax.legend_.get_texts():
                entry.set_fontsize(fs)
        for entry in ax.texts:
            entry.set_fontsize(fs)
        for entry in ax.tables:
            entry.set_fontsize(fs)

labels_time = [2019, 2020, 2025, 2030, 2035, 2040, 2045, 2050]
plt.rcParams['axes.xmargin'] = 0
plt.rcParams['axes.ymargin'] = 0
plt.rcParams['legend.frameon'] = False

def plot_outcomes(exp,out,out_to_show,
    group_by= None,
    density=None,title=None,
    exp_to_show = None,
    grouping_specifiers = None,
    legend = False,
    div_by_thousand = False,
    div_by_million = False,
    div_by_billion = False,
    zero = False ,
    yupperlim = None,
    xlabel = False,
    ylabel = False,
    alpha = None,
    sizex = None,
    sizey = None):
    out_copy = out
    if div_by_thousand == True:
        out_copy[title] = out_copy[out_to_show]/1e3
        lines(experiments = exp, outcomes = out_copy,
              experiments_to_show = exp_to_show,
              outcomes_to_show = title, legend = legend,
              group_by = group_by, density = density,
              grouping_specifiers = grouping_specifiers)
    elif div_by_million == True:
        out_copy[title] = out_copy[out_to_show]/1e6
        lines(experiments = exp, outcomes = out_copy,
              experiments_to_show = exp_to_show,
              outcomes_to_show = title,legend = legend,
              group_by = group_by, density = density,
              grouping_specifiers = grouping_specifiers)
    elif div_by_billion == True:
        out_copy[title] = out_copy[out_to_show]/1e9
        lines(experiments = exp, outcomes = out_copy,
              experiments_to_show = exp_to_show,
              outcomes_to_show = title,legend = legend,
              group_by = group_by, density = density,
              grouping_specifiers = grouping_specifiers)
    else:
        out_copy[title] = out_copy[out_to_show]
        lines(experiments = exp, outcomes = out_copy,
              experiments_to_show = exp_to_show,
              outcomes_to_show = title,legend = legend,
              group_by = group_by, density = density,
              grouping_specifiers = grouping_specifiers)
    fig = plt.gcf()
    fig.set_size_inches(6,3)
    if sizex:
        fig.set_size_inches(sizex,sizey)
    ax = fig.get_axes()
    if zero == True:
        ax[0].set_ylim([0,yupperlim])
    ax[0].set_xticklabels(labels_time)
    if xlabel:
        ax[0].set(xlabel=xlabel)
    if ylabel:
        ax[0].set(ylabel=ylabel)
    if alpha:
        for line in ax[0].get_lines():
            line.set_alpha(alpha)
    short_title = title.replace(" ","")
    change_fontsize(fig)
    sns.despine()
    save_fig(fig,wd+'\\plots\\visual_analysis',short_title)
    plt.show()

        
def make_clusters(exp,out,out_to_show,ref_year):
    nr_exp = len(exp)
    cluster_col = np.zeros(nr_exp, dtype=int)
    cluster_col_df = pd.Series(cluster_col)
    exp_copy = exp.copy()
    exp_copy['cluster'] = cluster_col_df
    index_cluster = exp_copy.columns.get_loc('cluster')
    
    col = out[out_to_show]
    i_col = np.int64((ref_year-2019)/0.0625)
    col_ryear = col[:,i_col]
    p90 = np.percentile(col_ryear,90)
    p80 = np.percentile(col_ryear,80)
    p70 = np.percentile(col_ryear,70)
    p60 = np.percentile(col_ryear,60)
    p50 = np.percentile(col_ryear,50)
    p40 = np.percentile(col_ryear,40)
    p30 = np.percentile(col_ryear,30)
    p20 = np.percentile(col_ryear,20)
    p10 = np.percentile(col_ryear,10)
    for i in range(0,nr_exp):
        if col[i,i_col] > p90:
            exp_copy.iloc[i,index_cluster] = np.int64(1)
            
        elif col[i,i_col] < p90 and col[i,i_col] > p80:
            exp_copy.iloc[i,index_cluster] = np.int64(2)
            
        elif col[i,i_col] < p80 and col[i,i_col] > p70:
            exp_copy.iloc[i,index_cluster] = np.int64(3)
            
        elif col[i,i_col] < p70 and col[i,i_col] > p60:
            exp_copy.iloc[i,index_cluster] = np.int64(4)
            
        elif col[i,i_col] < p60 and col[i,i_col] > p50:
            exp_copy.iloc[i,index_cluster] = np.int64(5)
            
        elif col[i,i_col] < p50 and col[i,i_col] > p40:
            exp_copy.iloc[i,index_cluster] = np.int64(6)
            
        elif col[i,i_col] < p40 and col[i,i_col] > p30:
            exp_copy.iloc[i,index_cluster] = np.int64(7)
            
        elif col[i,i_col] < p30 and col[i,i_col] > p20:
            exp_copy.iloc[i,index_cluster] = np.int64(8)
            
        elif col[i,i_col] < p20 and col[i,i_col] > p10:
            exp_copy.iloc[i,index_cluster] = np.int64(9)
            
        elif col[i,i_col] < p10: 
            exp_copy.iloc[i,index_cluster] = np.int64(10)
            
    return exp_copy
        

exp_cum_emissions_clustered = make_clusters(experiments,outcomes,'Cumulative emissions in tonnes',2050)
exp_pairplot = exp_cum_emissions_clustered.copy()

clusters = exp_cum_emissions_clustered['cluster']
for i, cluster in enumerate(np.unique(clusters)):
    exp_pairplot.loc[clusters==cluster, 'cluster'] = str(i)

#emissions_time_series = outcomes.get("Total emissions cracking in tonnes")
#data = pd.DataFrame(emissions_time_series[:,496])

#Generate scatterplot
KPI_col1 = outcomes.get('Cum. emissions 2050')
KPI_col2 = outcomes.get('Cum. production costs 2050')
KPI_col3 = outcomes.get('Cum. policy costs 2050')
KPI_array = np.array([KPI_col1,KPI_col2,KPI_col3])
trans_KPI_array = np.transpose(KPI_array)
data = pd.DataFrame(trans_KPI_array, columns=['Cum. emissions','Cum. prod. costs', 'Cum. pol. costs'])
data['cluster'] = clusters

variables = list(['Cum. emissions','Cum. prod. costs', 'Cum. pol. costs'])
#sns.pairplot(data, hue='policy')  # vars=variables[1:]
sns.pairplot(data, hue='cluster', vars=variables)
plt.savefig('snspairplot_kpis_500s_100p',dpi=300)
plt.show()

# #Generate clustered outcomes
# plot_outcomes(exp_cum_emissions_clustered,outcomes,out_to_show = 'Cumulative emissions in tonnes', 
#                 group_by='cluster', grouping_specifiers = {'p90':1, 'p80': 2, 'p70': 3, 'p60': 4 , 'p50': 5, 'p40': 6, 'p30': 7, 'p20': 8, 'p10': 9, 'p0': 10}, 
#                 density = Density.KDE, div_by_million=True,
#                 title = 'Cumulative emissions in megatonnes', 
#                 zero = True, xlabel = 'time [year]', ylabel = 'emissions [Mtonne CO2]', alpha = 0.05)

# # exp_cum_prodcosts_clustered = make_clusters(experiments,outcomes,'Cumulative production costs',2050)
# plot_outcomes(exp_cum_emissions_clustered,outcomes,out_to_show = 'Cumulative production costs', 
#                 group_by='cluster', grouping_specifiers = {'p90':1, 'p80': 2, 'p70': 3, 'p60': 4 , 'p50': 5, 'p40': 6, 'p30': 7, 'p20': 8, 'p10': 9, 'p0': 10}, 
#                 density = Density.KDE, div_by_billion = True,
#                 title = 'Cumulative production costs in billion euros', 
#                 xlabel = 'time [year]', ylabel = 'costs [Beuro]', alpha = 0.05)  

# # # exp_cum_polcosts_clustered = make_clusters(experiments,outcomes,'Cumulative policy costs',2050)
# plot_outcomes(exp_cum_emissions_clustered,outcomes,out_to_show = 'Cumulative policy costs', 
#                 group_by='cluster', grouping_specifiers = {'p90':1, 'p80': 2, 'p70': 3, 'p60': 4 , 'p50': 5, 'p40': 6, 'p30': 7, 'p20': 8, 'p10': 9, 'p0': 10}, 
#                 density = Density.KDE, div_by_billion=True,
#                 title = 'Cumulative policy costs in billion euros', 
#                 xlabel = 'time [year]', ylabel = 'costs [Beuro]', alpha = 0.05)    

# exp_emissions_clustered = make_clusters(experiments,outcomes,'Total emissions cracking in tonnes',2050)

# plot_outcomes(exp_emissions_clustered,outcomes,out_to_show = 'Total emissions cracking in tonnes', 
#                 group_by='cluster', grouping_specifiers = {'p90':1, 'p80': 2, 'p70': 3, 'p60': 4 , 'p50': 5, 'p40': 6, 'p30': 7, 'p20': 8, 'p10': 9, 'p0': 10}, 
#                 density = Density.KDE, div_by_thousand=True,
#                 title = 'Annual CO2 emissions cracking in kilotonnes', 
#                 zero = True, yupperlim = 1900, xlabel = 'time [year]', ylabel = 'emissions [ktonne/year]', alpha = 0.05)

# # # exp_prodcosts_clustered = make_clusters(experiments,outcomes,'Production costs per tonne ethylene',2050)
# plot_outcomes(exp_emissions_clustered,outcomes,out_to_show = 'Production costs per tonne ethylene', 
#                 group_by='cluster', grouping_specifiers = {'p90':1, 'p80': 2, 'p70': 3, 'p60': 4 , 'p50': 5, 'p40': 6, 'p30': 7, 'p20': 8, 'p10': 9, 'p0': 10}, 
#                 density = Density.KDE, 
#                 title = 'Production costs per tonne ethylene', 
#                 xlabel = 'time [year]', ylabel = 'costs [euro/tonne]', alpha = 0.05)  

# # # exp_polcosts_clustered = make_clusters(experiments,outcomes,'Total policy costs per year',2050)
# plot_outcomes(exp_emissions_clustered,outcomes,out_to_show = 'Total policy costs per year', 
#                 group_by='cluster', grouping_specifiers = {'p90':1, 'p80': 2, 'p70': 3, 'p60': 4 , 'p50': 5, 'p40': 6, 'p30': 7, 'p20': 8, 'p10': 9, 'p0': 10}, 
#                 density = Density.KDE, div_by_million=True,
#                 title = 'Annual costs of combined policy options', 
#                 xlabel = 'time [year]', ylabel = 'costs [Meuro/year]', alpha = 0.05)

# #Unclustered outcomes for the KDE plots
# plot_outcomes(experiments,outcomes,out_to_show = 'Total emissions cracking in tonnes', density = Density.KDE, div_by_thousand=True,
#                 title = 'Annual CO2 cracking in kilotonnes', zero = True, yupperlim = 1900, xlabel = 'time [year]', ylabel = 'emissions [ktonne/year]', alpha = 0.05)

# plot_outcomes(experiments,outcomes,out_to_show = 'Production costs per tonne ethylene', density = Density.KDE,
#                 title = 'Production costs per tonne ethylene', xlabel = 'time [year]', ylabel = 'costs [euro/tonne]', alpha = 0.05)

# plot_outcomes(experiments,outcomes,out_to_show = 'Total policy costs per year', density = Density.KDE, div_by_million=True,
#                 title = 'Annual costs of combined policy options', xlabel = 'time [year]', ylabel = 'costs [Meuro/year]', alpha = 0.05)


# plot_outcomes(experiments,outcomes,out_to_show = 'Cumulative emissions in tonnes', 
#                 density = Density.KDE, div_by_million=True,
#                 title = 'Cumulative emissions in megatonnes', 
#                 zero = True, xlabel = 'time [year]', ylabel = 'emissions [Mtonne CO2]', alpha = 0.05)

# # exp_cum_prodcosts_clustered = make_clusters(experiments,outcomes,'Cumulative production costs',2050)
# plot_outcomes(experiments,outcomes,out_to_show = 'Cumulative production costs', 
#                 density = Density.KDE, div_by_billion = True,
#                 title = 'Cumulative production costs in billion euros', 
#                 xlabel = 'time [year]', ylabel = 'costs [Beuro]', alpha = 0.05)  

# # # exp_cum_polcosts_clustered = make_clusters(experiments,outcomes,'Cumulative policy costs',2050)
# plot_outcomes(experiments,outcomes,out_to_show = 'Cumulative policy costs', 
#                 density = Density.KDE, div_by_billion=True,
#                 title = 'Cumulative policy costs in billion euros', 
#                 xlabel = 'time [year]', ylabel = 'costs [Beuro]', alpha = 0.05)   
# #lines(experiments,outcomes,outcomes_to_show = 'Total emissions cracking in tonnes', density = Density.KDE)

# # plot_outcomes(experiments,outcomes,out_to_show = 'Total electricity consumption PJ per year', density = Density.KDE,
# #                 title = 'Total electricity consumption PJ per year', xlabel = 'time [year]', ylabel = 'elec. cons. [PJ]', alpha = 0.05)

# # plot_outcomes(experiments,outcomes,out_to_show = 'Total gas consumption PJ per year', density = Density.KDE,
# #                 title = 'Total gas consumption PJ per year', xlabel = 'time [year]', ylabel = 'gas cons. [PJ]', alpha = 0.05)

# # plot_outcomes(experiments,outcomes,out_to_show = 'Percentage of electric furnaces', density = Density.KDE,
# #                 title = 'Percentage of electric furnaces', xlabel = 'time [year]', ylabel = '[%]', alpha = 0.05)

# # plot_outcomes(experiments,outcomes,out_to_show = 'Percentage of electric compressors', density = Density.KDE,
# #                 title = 'Percentage of electric compressors', xlabel = 'time [year]', ylabel = '[%]', alpha = 0.05)

# # plot_outcomes(experiments,outcomes,out_to_show = 'Percentage of electric boiler capacity', density = Density.KDE,
# #                 title = 'Percentage of electric boiler capacity', xlabel = 'time [year]', ylabel = '[%]', alpha = 0.05)

# # plot_outcomes(experiments,outcomes,out_to_show = 'Ratio of costs electric over conventional furnace', density = Density.KDE,
# #                 title = 'Ratio of costs electric over conventional furnace', xlabel = 'time [year]', ylabel = '[-]', alpha = 0.05)

# # plot_outcomes(experiments,outcomes,out_to_show = 'Ratio of costs electric over conventional compressor', density = Density.KDE,
# #                 title = 'Ratio of costs electric over conventional compressor', xlabel = 'time [year]', ylabel = '[-]', alpha = 0.05)

# # plot_outcomes(experiments,outcomes,out_to_show = 'Ratio of costs electric over conventional boiler', density = Density.KDE,
# #                 title = 'Ratio of costs electric over conventional boiler', xlabel = 'time [year]', ylabel = '[-]', alpha = 0.05)

# # plot_outcomes(experiments,outcomes,out_to_show = 'Ratio of costs hybrid over conventional boiler', density = Density.KDE,
# #                 title = 'Ratio of costs hybrid over conventional boiler', xlabel = 'time [year]', ylabel = '[-]', alpha = 0.05)

# # plot_outcomes(experiments,outcomes,out_to_show = '"Output capacity conventional boiler total (incl hybrid)"', density = Density.KDE,
# #                 title = 'Output capacity conventional boiler total (incl hybrid)', xlabel = 'time [year]', ylabel = 'capacity [MW]', alpha = 0.05)

# # plot_outcomes(experiments,outcomes,out_to_show = '"Output capacity electric boiler total (incl hybrid)"', density = Density.KDE,
# #                 title = 'Output capacity electric boiler total (incl hybrid)', xlabel = 'time [year]', ylabel = 'capacity [MW]', alpha = 0.05)

# # plot_outcomes(experiments,outcomes,out_to_show = 'Utilized output capacity hybrid boiler conventional', density = Density.KDE,
# #                 title = 'Utilized output capacity hybrid boiler conventional', xlabel = 'time [year]', ylabel = 'capacity [MW]', alpha = 0.05)

# # plot_outcomes(experiments,outcomes,out_to_show = 'Utilized output capacity hybrid boiler electric', density = Density.KDE,
# #                 title = 'Utilized output capacity hybrid boiler electric', xlabel = 'time [year]', ylabel = 'capacity [MW]', alpha = 0.05)

# # plot_outcomes(experiments,outcomes,out_to_show = 'Capacity boiler conventional', density = Density.KDE,
# #                 title = 'Capacity boiler conventional', xlabel = 'time [year]', ylabel = 'capacity [MW]', alpha = 0.05)

# # plot_outcomes(experiments,outcomes,out_to_show = 'Capacity boiler electric', density = Density.KDE,
# #                 title = 'Capacity boiler electric', xlabel = 'time [year]', ylabel = 'capacity [MW]', alpha = 0.05)

# # plot_outcomes(experiments,outcomes,out_to_show = 'Capacity boiler hybrid', density = Density.KDE,
# #                 title = 'Capacity boiler hybrid', xlabel = 'time [year]', ylabel = 'capacity [MW]', alpha = 0.05)

# # # nr_exp = len(experiments)
# # # cluster_col = np.zeros(nr_exp, dtype=int)
# # # cluster_col_df = pd.Series(cluster_col)
# # # exp_copy = experiments.copy()
# # # exp_copy['cluster'] = cluster_col_df
# # # index_cluster = exp_copy.columns.get_loc('cluster')

# # # emissions = outcomes['Total emissions cracking in tonnes']
# # # i_time = np.shape(emissions)[1]
# # # # emissions2 = pd.Dataframe(emissions)
# # # # emissions3 = emissions2.sort_values(by=[496])
# # # # emissions4 = np.array(emissions3)

# # # out_copy = outcomes.copy()
# # # for i in range(0,nr_exp):
# # #     if emissions[i,-1] > 500000:
# # #         exp_copy.iloc[i,index_cluster] = np.int64(1)
# # #     else:
# # #         exp_copy.iloc[i,index_cluster] = np.int64(0)    
