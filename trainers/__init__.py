from . import ClassificationTrainer, MLMTrainer, RegressionTrainer, NERTrainer

trainer = {
    'classification': ClassificationTrainer,
    'mlm': MLMTrainer,
    'regression': RegressionTrainer,
    'ner': NERTrainer,
}

def get_trainer(task: str):
    return trainer[task]
