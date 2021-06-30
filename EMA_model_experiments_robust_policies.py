# -*- coding: utf-8 -*-
"""
Created on Sun May 16 14:24:16 2021

@author: benjaminschoemaker
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 16:41:27 2021

@author: benjaminschoemaker
"""
import os 
import pandas as pd
import numpy as np
from ema_workbench import (TimeSeriesOutcome,
                           Policy, RealParameter, CategoricalParameter, ema_logging, ScalarOutcome, MultiprocessingEvaluator, save_results, perform_experiments)

from ema_workbench.connectors.vensim import VensimModel

def value_end(timeseries):
    return timeseries[-1]

def rank_data(df_orgnl,metric=None):
    df = df_orgnl.copy()
    if metric == 1:
        df_new = df.sort_values(by=['Max. cum. emissions'])
    
    elif metric==2:
        df_new = df.sort_values(by=['Dev. cum. emissions'])
        
    elif metric==3:
        df_new = df.sort_values(by=['Peakedness cum. emissions'])
        
    return df_new

def results_subsets(results,subset_size):
    results_collection = {}
    counter = 0 
    i = 0 
    while counter < len(results):
        if counter+subset_size < len(results):
            results_collection[i] = results[counter:counter+subset_size]
        else:
            results_collection[i] = results[counter:len(results)]
        counter += subset_size
        i += 1
    return results_collection

# def prepare_results_collection(results_collection,results):
#     index = results.columns.get_loc('DPR switch')
#     results_collection_new = results_collection.copy()
#     for i in range(0,len(results_collection)):
#         for j in range (0,len(results_collection[i])):
#             if results_collection[i].iloc[j,index] == "Category('0', 0)":
#                 results_collection_new[i].iloc[j,index] = np.int64(0)
#             else:
#                 results_collection_new[i].iloc[j,index] = np.int64(1)

def change_dpr_format(results):
    index = results.columns.get_loc('DPR switch')
    for i in range (0,len(results)):
        if results.iloc[i,index] == "Category('0', 0)":
            results.iloc[i,index] = np.int64(0)
        else:
            results.iloc[i,index] = np.int64(1)
                
robust_results1 = pd.read_csv('output_data\\robust_results1.csv').drop(columns='Unnamed: 0') 
robust_results2 = pd.read_csv('output_data\\robust_2000\\robust_results2.csv').drop(columns='Unnamed: 0') 
robust_results3 = pd.read_csv('output_data\\robust_2000\\robust_results3.csv').drop(columns='Unnamed: 0') 

robust_results1_1 = rank_data(robust_results1,metric=1)
robust_results2_1 = rank_data(robust_results2,metric=2)
robust_results3_1 = rank_data(robust_results3,metric=3)

change_dpr_format(robust_results1_1)
change_dpr_format(robust_results2_1)
change_dpr_format(robust_results3_1)

robust_col1 = results_subsets(robust_results1_1,23)
robust_col2 = results_subsets(robust_results2_1,20)
robust_col3 = results_subsets(robust_results3_1,24)

            
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
                                  RealParameter("Maximum percentage of boilers to become hybrid per year", 0, 2.5)]
                                  #RealParameter("Initial gas demand other processes", 0, 25)
                                 
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
    
    #vensimModel.outcomes = [TimeSeriesOutcome('Total emissions cracking in tonnes')]
    vensimModel.outcomes = [ScalarOutcome('Cum. emissions 2050', variable_name='Cumulative emissions in tonnes', 
                                          function=value_end),
                            ScalarOutcome('Cum. production costs 2050', variable_name='Cumulative production costs', 
                                          function=value_end),
                            ScalarOutcome('Cum. policy costs 2050', variable_name='Cumulative policy costs', 
                                          function=value_end),
                            TimeSeriesOutcome('Total emissions cracking in tonnes'),
                            TimeSeriesOutcome('Production costs per tonne ethylene'),
                            TimeSeriesOutcome('Total policy costs per year'),
                            # ScalarOutcome('Emissions 2050', variable_name='Total emissions cracking in tonnes', 
                            #               function=value_end),
                            TimeSeriesOutcome('Cumulative emissions in tonnes'),
                            TimeSeriesOutcome('Cumulative production costs'),
                            TimeSeriesOutcome('Cumulative policy costs')] 
                            # TimeSeriesOutcome('Total electricity consumption PJ per year' ),
                            # TimeSeriesOutcome('Total gas consumption PJ per year' ),
                            # TimeSeriesOutcome('Percentage of electric furnaces'),
                            # TimeSeriesOutcome('Percentage of electric compressors' ),
                            # TimeSeriesOutcome('Percentage of electric boiler capacity' ),
                            # TimeSeriesOutcome('Ratio of costs electric over conventional furnace' ),
                            # TimeSeriesOutcome('Ratio of costs electric over conventional compressor' ),
                            # TimeSeriesOutcome('Ratio of costs electric over conventional boiler' ),
                            # TimeSeriesOutcome('Ratio of costs hybrid over conventional boiler' ),
                            # TimeSeriesOutcome('"Output capacity conventional boiler total (incl hybrid)"'),
                            # TimeSeriesOutcome('"Output capacity electric boiler total (incl hybrid)"'),
                            # TimeSeriesOutcome('Utilized output capacity hybrid boiler conventional'),
                            # TimeSeriesOutcome('Utilized output capacity hybrid boiler electric'),
                            # TimeSeriesOutcome('Capacity boiler conventional'),
                            # TimeSeriesOutcome('Capacity boiler electric'),
                            # TimeSeriesOutcome('Capacity boiler hybrid')]

    #robust_results_1 = pd.read_csv('output_data\\robust_results1.csv').drop(columns='Unnamed: 0') 
    #robust_policies_1 = robust_results_1.drop(columns=['Max. cum. emissions','Max. cum. production costs', 'Max. cum. policy costs'])
    
    robust_policies = [
        Policy('0', **{k:v for k, v in robust_col1[0].iloc[0].items()
                        if k in vensimModel.levers}),
                Policy('1', **{k:v for k, v in robust_col1[4].iloc[0].items()
                        if k in vensimModel.levers}),
                Policy('2', **{k:v for k, v in robust_col1[5].iloc[0].items()
                        if k in vensimModel.levers}),
                Policy('3', **{k:v for k, v in robust_col1[7].iloc[0].items()
                        if k in vensimModel.levers}),
                Policy('4', **{k:v for k, v in robust_col1[8].iloc[0].items()
                        if k in vensimModel.levers}),
                Policy('5', **{k:v for k, v in robust_col2[0].iloc[0].items()
                        if k in vensimModel.levers}),
                Policy('6', **{k:v for k, v in robust_col2[4].iloc[0].items()
                        if k in vensimModel.levers}),
                Policy('7', **{k:v for k, v in robust_col2[8].iloc[0].items()
                        if k in vensimModel.levers}),
                Policy('8', **{k:v for k, v in robust_col2[12].iloc[0].items()
                        if k in vensimModel.levers}),
                Policy('9', **{k:v for k, v in robust_col2[17].iloc[0].items()
                        if k in vensimModel.levers}),
                Policy('10', **{k:v for k, v in robust_col2[18].iloc[0].items()
                        if k in vensimModel.levers}),
                Policy('11', **{k:v for k, v in robust_col2[20].iloc[0].items()
                        if k in vensimModel.levers}),
                Policy('12', **{k:v for k, v in robust_col2[24].iloc[0].items()
                        if k in vensimModel.levers}),
                Policy('13', **{k:v for k, v in robust_col2[28].iloc[0].items()
                        if k in vensimModel.levers}),
                Policy('14', **{k:v for k, v in robust_col2[31].iloc[0].items()
                        if k in vensimModel.levers}),
                Policy('15', **{k:v for k, v in robust_col2[38].iloc[0].items()
                        if k in vensimModel.levers}),
                Policy('16', **{k:v for k, v in robust_col3[0].iloc[0].items()
                        if k in vensimModel.levers}),
                Policy('17', **{k:v for k, v in robust_col3[2].iloc[0].items()
                        if k in vensimModel.levers}),
                Policy('18', **{k:v for k, v in robust_col3[4].iloc[0].items()
                        if k in vensimModel.levers}),
                Policy('19', **{k:v for k, v in robust_col3[8].iloc[0].items()
                        if k in vensimModel.levers}),
                Policy('20', **{k:v for k, v in robust_col3[14].iloc[0].items()
                        if k in vensimModel.levers})] 
    
    
    with MultiprocessingEvaluator(vensimModel) as evaluator:
        results = evaluator.perform_experiments(scenarios=300,policies=robust_policies)
    
    # experiments, outcomes = results
    
    save_results(results, 'output_data\\robust_results_300scenarios_4.tar.gz')
    
    # policies = [Policy('no policy',
    #                    model_file='FLUvensimV1basecase.vpm'),
    #             Policy('static policy',
    #                    model_file='FLUvensimV1static.vpm'),
    #             Policy('adaptive policy',
    #                    model_file='FLUvensimV1dynamic.vpm')
    #             ]
    
    #policies = [Policy('test', to_list(robust_policies_1
    #reference = Scenario('reference', b=0.4, q=2, mean=0.02, stdev=0.01)
    
    
                                 
    #save_results(results, 'output_data\\results_300scenarios_100policies.tar.gz')
    
    # experiments, outcomes = results
    
    # policies = experiments['policy']
    # for i, policy in enumerate(np.unique(policies)):
    #     experiments.loc[policies==policy, 'policy'] = str(i)
    
    # #emissions_time_series = outcomes.get("Total emissions cracking in tonnes")
    # #data = pd.DataFrame(emissions_time_series[:,496])
    # KPI_col1 = outcomes.get('Emissions 2050')
    # KPI_col2 = outcomes.get('Cum. OPEX 2050')
    # KPI_col3 = outcomes.get('Cum. policy costs 2050')
    # KPI_array = np.array([KPI_col1,KPI_col2,KPI_col3])
    # trans_KPI_array = np.transpose(KPI_array)
    # data = pd.DataFrame(trans_KPI_array, columns=['Emissions 2050','Cum. OPEX 2050', 'Cum. policy costs 2050'])
    # data['policy'] = policies
    
    # variables = list(outcomes.keys())
    # #sns.pairplot(data, hue='policy')  # vars=variables[1:]
    # sns.pairplot(data, hue='policy', vars=variables[1:])
    # plt.show()

    