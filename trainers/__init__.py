from . import ClassificationTrainer, RegressionTrainer, NERTrainer

trainer = {
    'classification': ClassificationTrainer,
    'regression': RegressionTrainer,
    'ner': NERTrainer,
}

def get_trainer(task: str):
    return trainer[task]
