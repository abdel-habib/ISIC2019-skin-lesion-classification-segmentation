from experiments.ClassifierAttentionExperiment import ClassifierAttentionExperiment
from experiments.ClassifierExperiment import ClassifierExperiment

def getExperiment(experiment_name):
    '''
    Returns the specified experiment class based on the provided experiment_name.
    '''
    network_class = globals().get(experiment_name, None)
    if network_class is None:
        raise ValueError(f"Invalid experiment_name: {experiment_name}")
    return network_class

