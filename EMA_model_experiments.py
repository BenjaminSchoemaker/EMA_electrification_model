# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 16:41:27 2021

@author: benjaminschoemaker
"""
import os 
from ema_workbench import (TimeSeriesOutcome,
                           Policy, RealParameter, CategoricalParameter, ema_logging, ScalarOutcome, MultiprocessingEvaluator, save_results, perform_experiments)

from ema_workbench.connectors.vensim import VensimModel

def value_end(timeseries):
    return timeseries[-1]

if __name__ == "__main__":
    # turn on logging
    ema_logging.log_to_stderr(ema_logging.INFO)

    wd = os.getcwd()
    wd = wd+'\\model_folder'
    vensimModel = VensimModel("ElectrificationModel", wd=wd, model_file='formal_model_new_cleaned_v15.vpmx')
   
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
        RealParameter("Efficiency improvements in conventional technologies", 0, 1),
        RealParameter("Response speed to gas becoming residual", 5, 50),
        RealParameter("Response speed of investment share to costs ratio", 5, 50),
        RealParameter("Response speed of investment share to costs ratio (hybrid boiler)", 5, 50),
        RealParameter("Renewables share projection 2030", 40, 90), 
        RealParameter("Time interval between turnarounds", 5, 7), 
        RealParameter("Acquisition time hybrid boiler", 0.5, 1), 
        RealParameter("Economic lifetime boiler", 10, 20), 
        RealParameter("Economic lifetime compressor", 10, 20), 
        RealParameter("Economic lifetime furnace", 20, 30), 
        RealParameter("Maximum percentage of boilers to become hybrid per year", 0, 2.5)]
                                 
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
                            # TimeSeriesOutcome('Cumulative production costs'),
                            # TimeSeriesOutcome('Cumulative policy costs'), 
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
                            # TimeSeriesOutcome('Capacity boiler hybrid')
                            ]
    
    with MultiprocessingEvaluator(vensimModel) as evaluator:
        results = evaluator.perform_experiments(scenarios=300,policies=100)
    
                                 
    save_results(results, 'output_data\\results_300scenarios_100policies_new.tar.gz')
    
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

    