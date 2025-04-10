from ..data.simulation import Simulation, SimulationData, CoilConfig
from .base import BaseOptimizer

from typing import Callable
import numpy as np

from tqdm import trange, tqdm
from scipy.optimize import minimize, OptimizeResult, dual_annealing


class ScipyOptimizer(BaseOptimizer):
    """
    DummyOptimizer is a dummy optimizer that randomly samples coil configurations and returns the best one.
    """
    def __init__(self,
                 cost_function: Callable[[SimulationData], float],
                 direction: str = "minimize",
                 max_iter: int = 100) -> None:
        super().__init__(cost_function, direction)
        self.max_iter = max_iter
        
    def _vector_to_coil_config(self, vector: np.ndarray) -> CoilConfig:
        phase = vector[:8]
        amplitude = vector[8:]
        return CoilConfig(phase=phase, amplitude=amplitude)
        
    def optimize(self, simulation: Simulation):
        pbar = tqdm(total=self.max_iter)
        
        def cost_fn(vector: np.ndarray) -> float:
            coil_config = self._vector_to_coil_config(vector)
            simulation_data = simulation(coil_config)
            sign = 1 if self.direction == "minimize" else -1
            return sign*self.cost_function(simulation_data)
        
        def callback_fn(intermediate_result: OptimizeResult):
            pbar.update(1)
            pbar.set_postfix_str(f"Cost {intermediate_result.fun:.2f}")
            
        def callback_fn_dual_annealing(x, f, context):
            pbar.update(1)
            pbar.set_postfix_str(f"Cost {f:.2f}")
        
        initial_guess = np.concat([np.zeros(8), 5*np.ones(8)])
        
        result = minimize(cost_fn, initial_guess, method="Nelder-Mead", options={"maxiter": self.max_iter}, callback=callback_fn)
        #result = dual_annealing(cost_fn, bounds=[(0, 2*np.pi)]*8 + [(0, 10)]*8, maxiter=self.max_iter, callback=callback_fn_dual_annealing, no_local_search=True)
        pbar.close()
        
        best_coil_config = self._vector_to_coil_config(result.x)    
        best_cost = result.fun
        print(f"Best cost: {best_cost}")
        
        return best_coil_config