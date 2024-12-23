import torch
from fn_utils import SoftMax, accuracy
from tqdm.notebook import tqdm


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu' # Выбор ресурсов для расчета


def predict(model, loader):
    """Make model predictions on loader data.
    
    Args:
        loader (DataLoader): DataLoader with data for predictions.
        model (Model): Current model.
    
    Returns:
        tuple: (act_classes, pred_classes):
            act_classes (list): List of actual classes.
            pred_classes (list): List of predicted classes.
    """
    
    
    model.eval()
    
    act_classes = None
    pred_classes = None
    
    for images, labels in iter(loader):
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        output = model.forward(images)
        
        if act_classes is None:
            act_classes = labels.cpu()
            pred_classes = output.cpu()
        else:
            act_classes = torch.cat((act_classes, labels.cpu()))
            pred_classes = torch.cat((pred_classes, output.cpu()))
    
    model = None
    images = None
    labels = None
    output = None
    sm = None
    
    torch.cuda.empty_cache()
    
    return act_classes, pred_classes


def predict_sample(sample, model):
    """Make model prediction on image.
    
    Args:
        sample (Tensor): Tensor with sample.
        model (Model): Current model.
    
    Returns:
        output (int): indiex of predicted class.
    """
    
    model.eval()
        
    sample = sample.unsqueeze(0)
    
    image = sample.to(DEVICE)
        
    output = model.forward(image)
    
    model = None
    images = None
    
    torch.cuda.empty_cache()
    
    return SoftMax(output).argmax(dim=1).item()


def evaluate_model(targets, predictions, criterion, metric_fn):
    """Make model evaluation.
    
    Args:
        targets (Model): Ground truth labels.
        predictions (DataLoader): Predictions of model.
        criterion (callable): Function for loss calculation.
        metric_fn (callable): Function for metric calculation.
    
    Returns:
        tuple: (loss, metric):
            loss (float): Value of loss.
            metric (float): Value of metric.
    """
    out = criterion(predictions, targets), accuracy(targets, SoftMax(predictions).argmax(dim=1))
    
    return out


def train_classifier(model, train_loader, val_loader, optimizer,
                     criterion, verbose,
                     metric_fn, metric_name,
                     epochs, print_every, callbacks, lr_scheduler):
    """Make model prediction on image.
    
    Args:
        model (Model): Model for training.
        train_loader (DataLoader): loader with train data.
        val_loader (DataLoader): loader with validation data.
        optimizer (Optimizer): Optimizer. 
        criterion (callable): Function for loss calculation.
        metric_fn (callable): Function for metric calculation.
        metric_name (str): Name of metric.
        epochs (int): Number of epoches.
        print_every (int): Number of iteration for update statusbar.
        callbacks (list): List of callbacks
    
    Returns:
        history (dict): Dict of lists with train history.
    """
    
    history = {'Train loss':[], 'Train {}'.format(metric_name):[],
               'Val loss':[], 'Val {}'.format(metric_name):[]}
    
    if callbacks:
        for i in callbacks:
            i.start(history, model)
    
    for e in range(epochs):
        model.train()

        running_loss = 0
        running_metric = 0
        
        
        stop = False
        
        steps = 0
        
        if verbose:
            train_print = ''
            bar = tqdm(range(len(train_loader)), desc="Epoch {}/{}".format(e+1, epochs), postfix=train_print)
        
        for images, labels in iter(train_loader):
            steps += 1
            images, labels = images.to(DEVICE), labels.to(DEVICE)
    
            optimizer.zero_grad()
            output = model.forward(images)
            
            loss = criterion(output, labels)

            loss.backward()
                
            optimizer.step()
            
            with torch.no_grad():
                running_loss += loss.item()
                running_metric += metric_fn(labels.cpu(), SoftMax(output).argmax(dim=1).cpu())
            
            if steps % print_every == 0:
                if verbose:
                    train_print = "Train loss: {:.4f}, Train {}: {:.4f}".format(running_loss / steps,
                                                                                metric_name,
                                                                                running_metric / steps)
                    bar.postfix = train_print
                model.train()
            
            if steps != len(train_loader) and verbose:
                bar.update()

        with torch.no_grad():
            targets, predictions = predict(model, val_loader)
            val_loss, val_metric = evaluate_model(targets, predictions, criterion, metric_fn)

        if verbose:
            train_print = "Train loss: {:.4f}, Train {}: {:.4f}, Val loss: {:.4f}, Val {}: {:.4f}".format(
                running_loss / steps,
                metric_name,
                running_metric / steps,
                val_loss,
                metric_name,
                val_metric)
        
        history['Train loss'].append(running_loss / steps)
        history['Train {}'.format(metric_name)].append(float(running_metric / steps))
        history['Val loss'].append(val_loss.item())
        history['Val {}'.format(metric_name)].append(float(val_metric))
        
        if lr_scheduler:
            lr_scheduler.step(val_loss)
        
        if callbacks:
            for i in callbacks:
                state_text, state = i.step()
                if state_text and verbose:
                    train_print += ', ' + state_text
                if state:
                    stop = True
        if verbose:
            bar.postfix = train_print
            bar.update()
            bar.close()
        
        if stop:
            if callbacks:
                for i in callbacks:
                    i.stop()
            model = None
            images = None
            labels = None
            output = None
            loss = None
            
            torch.cuda.empty_cache()
            
            break
                        
        images = None
        labels = None
        output = None
        loss = None
        
        torch.cuda.empty_cache()
        

    if callbacks:
        for i in callbacks:
            i.stop()
    
    model = None
    
    torch.cuda.empty_cache()
    
    return history


def dict2str(dict1):
    out = str(dict1).replace("}", "")
    out = str(out).replace("{", "")
    out = str(out).replace("\"", "")
    out = str(out).replace("\'", "")
    out = str(out).replace(":", "")
    return out