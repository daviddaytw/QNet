import torch

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """

    #round predictions to the closest integer
    preds = preds.argmax(dim=1)
    correct = (preds == y).float() #convert into float for division
    acc = correct.sum() / len(correct)
    return acc


def train(model, dataloader, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.train()
    for label, text in dataloader:
        optimizer.zero_grad()

        predictions = model(text).squeeze(1)
        #label = label.unsqueeze(1)
        loss = criterion(predictions, label)
        #loss = F.nll_loss(predictions, label)
        acc = binary_accuracy(predictions, label)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(dataloader), epoch_acc / len(dataloader)


def evaluate(model, dataloader, criterion):

    epoch_loss = 0
    epoch_acc = 0

    model.eval()
    with torch.no_grad():
        for label, text in dataloader:
            predictions = model(text).squeeze(1)
            #label = label.unsqueeze(1)
            loss = criterion(predictions, label)
            #loss = F.nll_loss(predictions, label)
            acc = binary_accuracy(predictions, label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(dataloader), epoch_acc / len(dataloader)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def yield_tokens(tokenizer, data_iter):
    for _, text in data_iter:
        yield tokenizer(text)
