"""
Evaluation framework for classification results
"""


class EvalClassification:
    def __init__(self, preds, labels):
        self.preds = preds
        self.labels = labels
        self.eval()

    def eval(self):
        assert (False, 'not implemented yet')
