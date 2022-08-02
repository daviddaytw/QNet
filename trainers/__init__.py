from . import ClassificationTrainer, MLMTrainer, RegressionTrainer

trainer = {
    'classification': ClassificationTrainer,
    'mlm': MLMTrainer,
    'regression': RegressionTrainer,
}

def get_trainer(task: str):
    return trainer[task]
