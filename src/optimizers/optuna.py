from ..data.simulation import Simulation, SimulationData, CoilConfig
from .base import BaseOptimizer

from typing import Callable
import numpy as np

from tqdm import trange
import optuna
from optuna import Trial

class OptunaOptimizer(BaseOptimizer):
    """
    DummyOptimizer is a dummy optimizer that randomly samples coil configurations and returns the best one.
    """
    def __init__(self,
                 cost_function: Callable[[SimulationData], float],
                 max_iter: int = 3000) -> None:
        super().__init__(cost_function)
        self.max_iter = max_iter
        
    def _sample_coil_config(self, trial: Trial) -> CoilConfig:
        phase = [trial.suggest_float(f"phase_{i}", 0, 2*np.pi) for i in range(8)]
        amplitude = [trial.suggest_float(f"amplitude_{i}", 0, 2) for i in range(8)]
        return CoilConfig(phase=phase, amplitude=amplitude)
        
    def optimize(self, simulation: Simulation):
        
        def objective(trial: Trial) -> float:
            coil_config = self._sample_coil_config(trial)
            simulation_data = simulation(coil_config)
            return self.cost_function(simulation_data)
        
        study = optuna.create_study(direction=self.direction, sampler=optuna.samplers.CmaEsSampler())
        study.optimize(objective, n_trials=self.max_iter, n_jobs=32)
        
        best_coil_config = self._sample_coil_config(study.best_trial)
        best_cost = study.best_value
        print(f"Best cost: {best_cost}")
        
        return best_coil_config