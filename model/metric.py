import torch


def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)


def mean_squared_error(output, target):
    """
    Calculate Mean Squared Error (MSE) between model output and target
    """
    with torch.no_grad():
        return torch.mean((output - target) ** 2).item()


def mean_absolute_error(output, target):
    """
    Calculate Mean Absolute Error (MAE) between model output and target
    """
    with torch.no_grad():
        return torch.mean(torch.abs(output - target)).item()


def r_squared(output, target):
    """
    Calculate R² (coefficient of determination) between model output and target
    R² = 1 - (sum of squared residuals / total sum of squares)
    """
    with torch.no_grad():
        target_mean = torch.mean(target)
        ss_total = torch.sum((target - target_mean) ** 2)
        ss_residual = torch.sum((target - output) ** 2)
        
        # Handle edge case where all target values are the same
        if ss_total == 0:
            return 0
        
        return (1 - ss_residual / ss_total).item()


def explained_variance(output, target):
    """
    Calculate explained variance between model output and target
    """
    with torch.no_grad():
        target_var = torch.var(target, unbiased=False)
        
        # Handle edge case where variance is zero
        if target_var == 0:
            return 0
            
        return (1 - torch.var(target - output, unbiased=False) / target_var).item()


def max_error(output, target):
    """
    Calculate maximum absolute error between model output and target
    """
    with torch.no_grad():
        return torch.max(torch.abs(output - target)).item()
