from spotpy.objectivefunctions import rms
from spotpy.parameter import Uniform
import numpy as np
import main_parameter_tuning_interface as sim


class spot_setup(object):
    grains_k_strach = Uniform(low=300, hight=500, optguess=400)
    
    def __init__(self, obj_func=None):
        self.obj_func = obj_func
        
    def simulation(self, x):
        grain_weight = sim.main(1200, run_simu=True, make_graphs=False, grains_k_strach=x[0])
        
        return grain_weight
        
    def evaluation(self):
        return [2.2]
        
    def objectivefunction(self, simulation, evaluation, params=None):
        if not self.obj_func:
            like = rmse(evaluation, simulation)
        else:
            like = self.obj_func(evalution, simulation)
            
        return like
        
        
        
if __name__ == "__main__":
    spot_agent = spot_setup()
    
    rep = 3
    
    sampler = spotpy.algorithms.fast(
        spot_setup, dbname="grains_k_strach_tuning", dbformat="csv", db_precision=np.float32
    )
    
    sampler.sample(rep)
    
    # Load the results gained with the fast sampler, stored in FAST_hymod.csv
    results = spotpy.analyser.load_csv_results("grains_k_strach_tuning")

    # Example plot to show the sensitivity index of each parameter
    spotpy.analyser.plot_fast_sensitivity(results, number_of_sensitiv_pars=3)

    # Example to get the sensitivity index of each parameter
    SI = spotpy.analyser.get_sensitivity_of_fast(results)
    
    