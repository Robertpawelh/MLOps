from torchmetrics.classification import F1Score, Accuracy, Precision, Recall
import torch.nn as nn

# TODO: connect metrics with DVC
class BinaryMetrics(nn.Module):
    def __init__(self, device: str, output_dim: int = 2):
        super().__init__()

        metrics = {'f1': F1Score,
                   'accuracy': Accuracy,
                   'precision': Precision,
                   'recall': Recall}
        metrics_dict = {name: metric(task='binary', num_classes=output_dim, average='macro') for name, metric in metrics.items()}
        self.metrics = nn.ModuleDict(metrics_dict)

    @property
    def device(self):
        return next(self.children()).device

    def calculate(self, probs, y_true, time):
        results = {metric_name: metric(probs, y_true) for metric_name, metric in self.metrics.items()}
        results['prediction_time'] = time

        return results

    def log_metrics(self, logger, loss, avg_metrics, current_epoch, metric_prefix):
        logger.log("epoch", current_epoch, on_epoch=True, on_step=False)
        logger.log(f"{metric_prefix}_loss", loss.item(), on_epoch=True, on_step=False)
        for k, v in avg_metrics.items():
            logger.log(f"{metric_prefix}_{k}_avg", v, on_epoch=True, on_step=False)
