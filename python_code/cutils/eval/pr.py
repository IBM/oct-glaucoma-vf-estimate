import colorsys
import numpy as np
from PIL import Image
from sklearn.metrics import precision_recall_curve

from cutils.viz.vizutils import figure2image


class PRCurves:
    """
    Generates the plot for  the precision recall curve
    """

    def __init__(self):
        self.data = []

    def add(self, gt, pred, legend):
        precision, recall, thresholds = precision_recall_curve(gt, pred)
        self.data.append([precision, recall, thresholds, legend])

    def generate(self):
        import matplotlib.pyplot as plt
        plt.clf()
        lw = 2

        N_class = len(self.data)
        half = int(np.ceil(N_class / 2.0))

        colors_half = (
            'b', 'g', 'r', 'c', 'm', 'y', 'k')  # [colorsys.hsv_to_rgb(x * 1.0 / half, 0.6, 1) for x in range(half)]
        colors = []
        colors.extend(colors_half)
        colors.extend(colors_half)

        lst = []
        lst_half1 = ['-' for c in colors_half]
        lst_half2 = [':' for c in colors_half]
        lst.extend(lst_half1)
        lst.extend(lst_half2)

        cc = 0
        plt.figure(figsize=(6, 6))
        l2d = None
        for pre, rec, th, legend in self.data:
            # roc_auc = auc(fpr, tpr)
            legend_disp = legend  # + ' ' + str(np.round(roc_auc, 2))
            l2d = plt.plot(pre, rec, lw=lw, color=colors[cc], linestyle=lst[cc],
                           label=legend_disp)
            cc += 1

        plt.xlabel('recall')
        plt.ylabel('precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        # plt.title('ROC curve')
        plt.legend(loc="lower right")

        im = figure2image(l2d[0].figure)
        print ('Size of roc image ' + str(np.shape(im)))
        return Image.fromarray(im)
