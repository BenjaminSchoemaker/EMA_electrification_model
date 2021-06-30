# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 16:46:52 2021

@author: benjaminschoemaker
"""


# more or less default imports when using
# the workbench
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from ema_workbench import (RealParameter, CategoricalParameter, ema_logging, ScalarOutcome, MultiprocessingEvaluator, save_results)

from ema_workbench.em_framework.optimization import (HyperVolume,
                                                     EpsilonProgress)

from ema_workbench.em_framework import sample_uncertainties

from ema_workbench.analysis import parcoords

from ema_workbench.connectors.vensim import VensimModel

def value_end(timeseries):
    return timeseries[-1]

if __name__ == "__main__":
    # turn on logging
    ema_logging.log_to_stderr(ema_logging.INFO)
    
    wd = os.getcwd()
    wd = wd+'\\model_folder'
    vensimModel = VensimModel("ElectrificationModel", wd=wd, model_file='formal_model_new_cleaned_v16.vpmx')
    
    vensimModel.uncertainties = [RealParameter("Electricity market price projection 2030", 33.3, 71.9),
                                 RealParameter("ETS carbon price projection 2050", 40, 1000),
                                 RealParameter("Gas market price projection 2030", 0.16, 0.319),
                                 RealParameter("Discount rate in percent", 10, 15),
                                 RealParameter("Retrofit investment cost per MW furnace", 300000, 1140000),
                                 RealParameter("Retrofit investment cost per MW compressor", 600000, 720000),
                                 RealParameter("Retrofit investment cost per MW boiler", 9710, 55000),
                                 RealParameter("Initial retrofit investment cost per MW furnace electric", 1140000, 3000000),
                                 RealParameter("Initial retrofit investment cost per MW compressor electric", 1000000, 2000000),
                                 RealParameter("Initial retrofit investment cost per MW boiler electric", 100000, 500000),
                                 RealParameter("Exogenous learning rate", 0, 0.00928),
                                 RealParameter("Export value of residual gas as a fraction of gas price", 0, 1),
                                 RealParameter("OPEX electric in percentage of investment costs", 1, 5),
                                 #RealParameter("Autonomous efficiency improvement", 0, 1),
                                 RealParameter("Efficiency improvements in conventional technologies", 0, 1),
                                 #RealParameter("Expected plant efficiency improvement 2050", 11, 22),
                                 RealParameter("Response speed to gas becoming residual", 5, 50),
                                 RealParameter("Response speed of investment share to costs ratio", 5, 50),
                                 RealParameter("Response speed of investment share to costs ratio (hybrid boiler)", 5, 50),
                                 RealParameter("Renewables share projection 2030", 40, 90), 
                                 RealParameter("Time interval between turnarounds", 5, 7), 
                                 RealParameter("Acquisition time hybrid boiler", 0.5, 1), 
                                 RealParameter("Economic lifetime boiler", 10, 20), 
                                 RealParameter("Economic lifetime compressor", 10, 20), 
                                 RealParameter("Economic lifetime furnace", 20, 30), 
                                 #RealParameter("Maximum boiler hybridisation per year", 5, 10), 
                                 RealParameter("Maximum percentage of boilers to become hybrid per year", 0, 2.5),
                                 #RealParameter("Initial gas demand other processes", 0, 25)
                                 ]
    vensimModel.levers = [RealParameter('"Base fee SDE++ per kWh"', 0.04, 0.14),
                          RealParameter("Carbon levy increase rate", 5.28, 21.12),
                          RealParameter("Carbon levy increase rate after 2030", 0, 21.12),
                          RealParameter("EHS grid connection costs", 1500000, 3000000),
                          RealParameter("Energy tax electricity", 0, 0.56),
                          RealParameter("Gas tax", 0.01281, 0.1281),
                          RealParameter("ODE electricity", 0, 0.4),
                          RealParameter("ODE gas", 0.0232, 0.232),
                          RealParameter("Variable transport tariff per kW per month furnace and compressor", 0, 1.23),
                          RealParameter("Variable transport tariff per kW per month boiler", 0, 2),
                          CategoricalParameter("DPR switch", (0, 1))]
    
    vensimModel.outcomes = [ScalarOutcome('Cum. emissions 2050', variable_name='Cumulative emissions in tonnes', 
                                          function=value_end),
                            ScalarOutcome('Cum. production costs 2050',  variable_name='Cumulative production costs', 
                                          function=value_end),
                            ScalarOutcome('Cum. policy costs 2050', variable_name='Cumulative policy costs', 
                                          function=value_end)]

    
    
    # vensimModel.outcomes = [TimeSeriesOutcome('Total emissions cracking in tonnes'), 
    #                         ScalarOutcome('Emissions in 2050', variable_name='Total emissions cracking in tonnes', function=value_end)]
    # results = perform_experiments(vensimModel,scenarios=10,policies=10)
    # experiments, outcomes = results
    
    

    MAXIMIZE = ScalarOutcome.MAXIMIZE
    MINIMIZE = ScalarOutcome.MINIMIZE
    
    #Metric 1: minimax
    
    # import functools
    
    # percentile10 = functools.partial(np.percentile, q=10)
    # percentile90 = functools.partial(np.percentile, q=90)
    
    robustness_functions_1 = [ScalarOutcome('Max. cum. emissions', 
                                            kind=MINIMIZE, variable_name='Cum. emissions 2050', 
                                          function=np.max),
                            ScalarOutcome('Max. cum. production costs', 
                                          kind=MINIMIZE, variable_name='Cum. production costs 2050', 
                                           function=np.max),
                            ScalarOutcome('Max. cum. policy costs', 
                                          kind=MINIMIZE, variable_name='Cum. policy costs 2050', 
                                          function=np.max)]
    #Metric 2: Undesirable deviations
    def dev_from_median_metric(outcome):
        score = 0
        median = np.median(outcome)
        subset_outcome = outcome[outcome > median]
        for i in range(0,len(subset_outcome)):
            score_i = (subset_outcome[i]-median)**2
            score += score_i
        return score
            
    robustness_functions_2 = [ScalarOutcome('Mean cum. emissions', 
                                            kind=MINIMIZE, variable_name='Cum. emissions 2050', 
                                          function=np.mean),
                              ScalarOutcome('Dev. cum. emissions', 
                                            kind=MINIMIZE, variable_name='Cum. emissions 2050', 
                                          function=dev_from_median_metric),
                              ScalarOutcome('Mean cum. production costs', 
                                            kind=MINIMIZE, variable_name='Cum. production costs 2050', 
                                           function=np.mean),
                              ScalarOutcome('Dev. cum. production costs', 
                                            kind=MINIMIZE, variable_name='Cum. production costs 2050', 
                                           function=dev_from_median_metric),
                              ScalarOutcome('Mean cum. policy costs', 
                                            kind=MINIMIZE, variable_name='Cum. policy costs 2050', 
                                          function=np.mean),
                              ScalarOutcome('Dev. cum. policy costs', 
                                            kind=MINIMIZE, variable_name='Cum. policy costs 2050', 
                                          function=dev_from_median_metric)]
    #Metric 3: Percentile-based peakedness
    def peakedness_metric(outcome):
        q90 = np.percentile(outcome,90)
        q10 = np.percentile(outcome,10)
        q75 = np.percentile(outcome,75)
        q25 = np.percentile(outcome,25)
        peakedness = (q90-q10)/(q75-q25)
        return peakedness
    
    robustness_functions_3 = [ScalarOutcome('Mean cum. emissions', 
                                            kind=MINIMIZE, variable_name='Cum. emissions 2050', 
                                          function=np.mean),
                              ScalarOutcome('Peakedness cum. emissions', 
                                            kind=MINIMIZE, variable_name='Cum. emissions 2050', 
                                          function=peakedness_metric),
                              ScalarOutcome('Mean cum. production costs', 
                                            kind=MINIMIZE, variable_name='Cum. production costs 2050', 
                                           function=np.mean),
                              ScalarOutcome('Peakedness cum. production costs', 
                                            kind=MINIMIZE, variable_name='Cum. production costs 2050', 
                                           function=peakedness_metric),
                              ScalarOutcome('Mean cum. policy costs', 
                                            kind=MINIMIZE, variable_name='Cum. policy costs 2050', 
                                          function=np.mean),
                              ScalarOutcome('Peakedness cum. policy costs', 
                                            kind=MINIMIZE, variable_name='Cum. policy costs 2050', 
                                          function=peakedness_metric)]
    
    # #Metric 4: Percentile-based skewness
    # def skewness_metric(outcome):
    #     q90 = np.percentile(outcome,90)
    #     q10 = np.percentile(outcome,10)
    #     q50 = np.median(outcome)
    #     skewness = ((q90+q10)/2-q50)/((q90-q10)/2)
    #     return skewness
    
    # robustness_functions_4 = [ScalarOutcome('Mean cum. emissions', kind=MINIMIZE, variable_name='Cum. emissions 2050', 
    #                                       function=np.mean),
    #                           ScalarOutcome('Skewness cum. emissions', kind=MINIMIZE, variable_name='Cum. emissions 2050', 
    #                                       function=skewness_metric),
    #                           ScalarOutcome('Mean cum. production costs', kind=MINIMIZE, variable_name='Cum. production costs 2050', 
    #                                        function=np.mean),
    #                           ScalarOutcome('Skewness cum. production costs', kind=MINIMIZE, variable_name='Cum. production costs 2050', 
    #                                        function=skewness_metric),
    #                           ScalarOutcome('Mean cum. policy costs', kind=MINIMIZE, variable_name='Cum. policy costs 2050', 
    #                                       function=np.mean),
    #                           ScalarOutcome('Skewness cum. policy costs', kind=MINIMIZE, variable_name='Cum. policy costs 2050', 
    #                                       function=skewness_metric)]
    
    #Specify convergence metrics
    
    
    convergence_metrics_1 = [HyperVolume(minimum=[0,0,-10**10],
                                         maximum=[10**8,10**11,10**10]), 
                                         EpsilonProgress()]
    convergence_metrics_2 = [HyperVolume(minimum=[0,0,0,0,-10**10,0],
                                         maximum=[10**8,10**15,10**11,10**21,10**10,10**20]), 
                                         EpsilonProgress()]
    convergence_metrics_3 = [HyperVolume(minimum=[0,0,0,0,-10**10,0],
                                         maximum=[10**8,10,10**11,10,10**10,10]), 
                                         EpsilonProgress()]
    
    #Define functions for tracking convergence
    def save_fig(fig, dir, name, dpi=300):
        fig.savefig('{}/fig_{}_{}dpi.png'.format(dir, name, dpi), dpi=dpi,
                bbox_inches='tight', format='png')
    
    def track_convergence(convergence,title):
        fig, (ax1, ax2) = plt.subplots(ncols=2, sharex=True, figsize=(8,4))
        ax1.plot(convergence.nfe, convergence.epsilon_progress)
        ax1.set_ylabel('$\epsilon$-progress')
        ax2.plot(convergence.nfe, convergence.hypervolume)
        ax2.set_ylabel('hypervolume')
        ax1.set_xlabel('number of function evaluations')
        ax2.set_xlabel('number of function evaluations')
        sns.despine()
        plt.show()
        save_fig(fig,wd,title)
        
    #Perform robust optimization

    n_scenarios = 50
    scenarios = sample_uncertainties(vensimModel, n_scenarios)
    
    # nfe = int(5000)
    
    # with MultiprocessingEvaluator(vensimModel) as evaluator:
    #     results_1, convergence_1 = evaluator.robust_optimize(robustness_functions_1, 
    #                                                          scenarios,
    #                         nfe=nfe, epsilons=[0.05,]*len(robustness_functions_1),
    #                         convergence=convergence_metrics_1)
    
    # results_1.to_csv('output_data\\robust_results1_new.csv')
    # convergence_1.to_csv('output_data\\robust_conv1_new.csv')
    # track_convergence(convergence_1,'track_convergence_1_new')
    
    nfe = int(2100)
    
    with MultiprocessingEvaluator(vensimModel) as evaluator:   
        results_2, convergence_2 = evaluator.robust_optimize(robustness_functions_2, 
                                                             scenarios,
                            nfe=nfe, epsilons=[0.05,]*len(robustness_functions_2),
                            convergence=convergence_metrics_2)
    
    results_2.to_csv('output_data\\robust_results2_new.csv')
    convergence_2.to_csv('output_data\\robust_conv2_new.csv')
    track_convergence(convergence_2,'track_convergence_2_new')
    
    with MultiprocessingEvaluator(vensimModel) as evaluator:
        results_3, convergence_3 = evaluator.robust_optimize(robustness_functions_3, 
                                                             scenarios,
                            nfe=nfe, epsilons=[0.05,]*len(robustness_functions_3),
                            convergence=convergence_metrics_3)
        
    results_3.to_csv('output_data\\robust_results3_new.csv')
    convergence_3.to_csv('output_data\\robust_conv3_new.csv')
    track_convergence(convergence_3,'track_convergence_3_new')
    
    # results_1.to_csv('output_data\\robust_results1_test.csv')
    # #results_1 = pd.read_csv('output_data\\robust_results1_test.csv').drop(columns='Unnamed: 0') 
    # ##results_1_1 = results_1.drop(columns='DPR switch')
    # results_2.to_csv('output_data\\robust_results2_test.csv')
    # #results_2 = pd.read_csv('output_data\\robust_results2_test.csv').drop(columns='Unnamed: 0') 
    # results_3.to_csv('output_data\\robust_results3_test.csv')
    # #results_3 = pd.read_csv('output_data\\robust_results3_test.csv').drop(columns='Unnamed: 0') 
    
    
    
    
        
    
        
    # results = optimize(vensimModel,nfe=10, searchover='levers',
    #                                   epsilons=[0.1,]*len(vensimModel.outcomes))
    
    # with MultiprocessingEvaluator(vensimModel) as evaluator:
    #     results = evaluator.optimize(nfe=10, searchover='levers',
    #                                   epsilons=[0.1,]*len(vensimModel.outcomes))
    
    # Example
    # def time_of_max(infected_fraction, time):
    #     index = np.where(infected_fraction == np.max(infected_fraction))
    #     timing = time[index][0]
    #     return timing


    # if _name_ == '_main_':
    #     ema_logging.log_to_stderr(ema_logging.INFO)
    
    #     model = VensimModel("fluCase", wd='./models/flu',
    #                         model_file='FLUvensimV1basecase.vpm')
    
    #     # outcomes
    #     model.outcomes = [TimeSeriesOutcome('deceased population region 1'),
    #                       TimeSeriesOutcome('infected fraction R1'),
    #                       ScalarOutcome('max infection fraction',
    #                                     variable_name='infected fraction R1',
    #                                     function=np.max),
    #                       ScalarOutcome('time of max',
    #                                     variable_name=[
    #                                         'infected fraction R1', 'TIME'],
    #                                     function=time_of_max)]
    
    #experiments = rf.drop_fields(experiments, [str(i) for i in range(100)], asrecarrary=True)
    
    #b = outcomes.get("Total emissions cracking in tonnes") # b is time series of first scenario
    
    
    # from ema_workbench import (MultiprocessingEvaluator, ema_logging,
    #                        perform_experiments)
    #ema_logging.log_to_stderr(ema_logging.INFO)
    
    # with MultiprocessingEvaluator(vensimModel) as evaluator:
    #     results = evaluator.perform_experiments(scenarios=10)

# import ema_workbench.analysis.pairs_plotting as pairs
# fig, axes = pairs.pairs_scatter(results) #doesn't work because it needs two instances
# plt.show()
    
# experiments, outcomes = results
# print(experiments.shape)
# print(list(outcomes.keys()))

# policies = experiments['policy']
# for i, policy in enumerate(np.unique(policies)):
#     experiments.loc[policies==policy, 'policy'] = str(i)

# data = pd.DataFrame(outcomes)
# data['policy'] = policies

# sns.pairplot(data, hue='policy', vars=list(outcomes.keys()))
# plt.show()