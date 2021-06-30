# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 11:02:24 2021

@author: benjaminschoemaker
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os


from ema_workbench import load_results

from ema_workbench.analysis import (feature_scoring, get_ex_feature_scores, RuleInductionType)

wd = os.getcwd()

def value_end(timeseries):
    return timeseries[-1]

results = load_results('output_data\\results_300scenarios_100policies.tar.gz')
    
experiments, outcomes = results

#Functions below adapted from master thesis Erika van der Linden (2020) https://repository.tudelft.nl/islandora/object/uuid%3Ae51dbb87-09f7-4c33-a956-226874a1e7b7

def save_fig(fig, dir, name, dpi=300):
    fig.savefig('{}/fig_{}_{}dpi.png'.format(dir, name, dpi), dpi=dpi,
                bbox_inches='tight', format='png')

def get_ex_feature_scores_top_factors (variable,top_nr):
    x= experiments.drop(['model', 'policy'], axis=1)
    y = outcomes[variable]
    fscores_time_list = []
    top_x = set()
    for i in range(2, y.shape[1], 16):
        data = y[:, i]
        fscores = get_ex_feature_scores(x, data,
        mode=RuleInductionType.REGRESSION)[0]
        top_x |= set(scores.nlargest(top_nr, 1).index.values)
        fscores_time_list.append(fscores)
    fscores_time = pd.concat(all_scores, axis=1, sort=False)
    fscores_time = fscores_time.loc[top_x, :]
    fscores_time.columns = np.arange(2019, 2050, 1)
    fscores_time = fscores_time.sort_values(by = [2019], ascending = False)
    return (fscores_time)

def plot_heatmap_overtime (fscores,title):
    sns.heatmap(fscores, cmap='viridis')
    fig = plt.gcf()
    ax = fig.get_axes()
    ax[0].set_xticklabels(np.arange(2019, 2051, 2))
    fig.autofmt_xdate()
    fig.set_size_inches(15,5)
    fig.suptitle('Extra trees feature scores for input parameters that most strongly influence '+title)
    shorttitle = title.replace(" ","")
    save_fig(fig,wd+'\\plots\\extra_trees','extra_trees'+shorttitle)
    plt.show()

#Feature scores over time
fscores_emissions = get_ex_feature_scores_top_factors('Total emissions cracking in tonnes',3)
plot_heatmap_overtime(fscores_emissions,title = 'CO2 emissions')

fscores_prod_costs = get_ex_feature_scores_top_factors('Production costs per tonne ethylene',3)
plot_heatmap_overtime(fscores_prod_costs,title = 'production costs per tonne ethylene')

fscores_pol_costs = get_ex_feature_scores_top_factors('Total policy costs per year',3)
plot_heatmap_overtime(fscores_pol_costs,title = 'policy costs')

#Extra trees feature scores for cumulative values
KPI_col1 = outcomes.get('Cum. emissions 2050')
KPI_col2 = outcomes.get('Cum. production costs 2050')
KPI_col3 = outcomes.get('Cum. policy costs 2050')
KPI_array = np.array([KPI_col1,KPI_col2,KPI_col3])
trans_KPI_array = np.transpose(KPI_array)
outcomes_KPI = pd.DataFrame(trans_KPI_array, columns=['Cum. emissions','Cum. prod. costs', 'Cum. pol. costs'])

fs = feature_scoring.get_feature_scores_all(experiments, outcomes_KPI)
fs2 = fs.drop('policy')
#sns.heatmap(fs2, cmap='viridis', annot=True)
sns.heatmap(fs2[0:17], cmap='viridis', annot=True)
plt.savefig('extra_trees_heatmap1', dpi=300, bbox_inches='tight')
sns.heatmap(fs2[17:35], cmap='viridis', annot=True)
plt.savefig('extra_trees_heatmap2', dpi=300, bbox_inches='tight')
plt.show()
