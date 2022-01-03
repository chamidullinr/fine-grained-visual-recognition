import torch
from torch import nn, optim

from src.core import training


def tune_temperature(model=None, validloader=None, *, device=None,
                     logits=None, targs=None, other_preds=None, verbose=True):
    """
    Tune the tempearature of the model using the validation set.

    Thanks to: https://github.com/gpleiss/temperature_scaling/blob/master/temperature_scaling.py
    """
    assert ((validloader is not None and model is not None) or 
            (logits is not None and targs is not None))

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = nn.NLLLoss()
    # criterion = nn.CrossEntropyLoss()

    # get labels and predictions in validloader
    if logits is None or targs is None:
        logits, targs, _ = training.predict(model, validloader, device=device)
    logits = torch.Tensor(logits).to(torch.float32).to(device)
    targs = torch.Tensor(targs).to(torch.long).to(device)
    if other_preds is not None:
        other_preds = torch.Tensor(other_preds).to(torch.float32).to(device)
    
    # calculate loss before temperature scaling
    loss_before = criterion(logits, targs).item()
    if verbose:
        print(f'Before temperature - NLL: {loss_before:.3f}')

    # optimize the temperature w.r.t. NLL
    temperature = nn.Parameter(torch.ones(1) * 1.5)
    optimizer = optim.LBFGS([temperature], lr=0.01, max_iter=50)

    def _eval():
        optimizer.zero_grad()
        # apply temperature scaling
        logits_ = logits / temperature.unsqueeze(1).expand(*logits.shape)
        preds = logits_.softmax(1)
        if other_preds is not None:
            preds = preds * other_preds
        loss = criterion(preds, targs)
        loss.backward()
        return loss
    optimizer.step(_eval)

    # calculate loss after temperature scaling
    logits_ = logits / temperature.unsqueeze(1).expand(*logits.shape)
    loss_after = criterion(logits_, targs).item()
    temperature = temperature.item()
    if verbose:
        print(f'Optimal temperature: {temperature:.3f}')
        print(f'After temperature - NLL: {loss_after:.3f}')

    return temperature
