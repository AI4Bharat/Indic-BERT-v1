

def mean_accuracy(preds, labels):
    return {'acc': (preds == labels).mean()}
