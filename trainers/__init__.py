from . import ClassificationTrainer, MLMTrainer

trainer = {
    'classification': ClassificationTrainer,
    'mlm': MLMTrainer,
}

def get_trainer(task: str):
    return trainer[task]
