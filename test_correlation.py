from torch.cuda.amp import autocast
import torch
import numpy as np

def test_correlation(model, test_generator, NUMBER_ITERATIONS):
    torch.cuda.empty_cache()
    model = model.cuda()

    labels = None
    iterations = 0
    for local_batch, local_labels in test_generator:
        with autocast():
            local_labels_pred = model(local_batch)
        if (labels == None):
            labels = local_labels
            pred_labels = local_labels_pred
        else:
            labels = torch.cat((labels, local_labels), 0)
            pred_labels = torch.cat((pred_labels, local_labels_pred), 0)
        
        torch.cuda.empty_cache()
        
        iterations += 1
        if (iterations >= NUMBER_ITERATIONS):
            break


    np_labels = labels.flatten().cpu().to(torch.float32).detach().numpy()
    np_pred_labels = pred_labels.flatten().cpu().to(torch.float32).detach().numpy()

    m, b = np.polyfit(np_labels, np_pred_labels, 1)
    print("Regression line: " + str(m) + "*X + " + str(b))

    print("Actual Mean: " + str(np.mean(np_labels)) + "; Actual Standard Deviation: " + str(np.std(np_labels)))
    print("Predicted Mean: " + str(np.mean(np_pred_labels)) + "; Predicted Standard Deviation: " + str(np.std(np_pred_labels))) 
