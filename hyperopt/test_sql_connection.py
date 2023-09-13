import importlib.util
import sys
from pytorch_lightning.loggers import TensorBoardLogger
import optuna
from optuna.integration import PyTorchLightningPruningCallback
# from ray import air, tune
# from ray.air import session
# from ray.tune import CLIReporter
from optuna.visualization import plot_contour
from optuna.visualization import plot_edf
from optuna.visualization import plot_intermediate_values
from optuna.visualization import plot_optimization_history
from optuna.visualization import plot_parallel_coordinate
from optuna.visualization import plot_param_importances
from optuna.visualization import plot_slice
# torch.multiprocessing.freeze_support()

experiment_name="testt"

study = optuna.create_study(
        study_name=experiment_name
        ,sampler=optuna.samplers.NSGAIISampler()    
        ,pruner=optuna.pruners.HyperbandPruner()
        ,storage="mysql://root@34.90.134.17/testt"
        ,load_if_exists=True
        #,storage="mysql://root@127.0.0.1:3306/picai_hp_35"
        )