import colorsys
import numpy as np
from PIL import Image

from cutils.viz.vizutils import figure2image
from .evalutils import dice_coefficient_batch, prec_recall


class EvalSegmentation:
    def __init__(self, preds, labels):
        """
        Evaluates the segmentation by computing dice coefficient, precision, recall,
        :param preds: NxHxWx1 prediction masks  where pixel values are between [0,1]
        :param labels: NxHxWx1 ground truth masks where pixel values are between [0,1]
        """
        assert (set(labels.flatten())) == set(
            np.asarray([0, 1])), ' Invalid  labels pixels value. The valid values are 0 and 1, found' + str(set(labels))

        self.preds = preds
        self.labels = labels
        self.evaluated = False
        self.eval()

    def eval(self):
        gt = np.asarray(self.labels)
        pred = np.asarray(self.preds)

        # the precision recall are not for each image, they are for all thresholds
        self.precision_all, self.recall_all, self.thresholds = prec_recall(gt, pred)
        min_index = np.argmin(np.abs(self.precision_all - self.recall_all))
        self.precision = self.precision_all[min_index]
        self.recall = self.recall_all[min_index]
        self.min_index=  min_index

        # print ' precision/recall at the operating threshold'+ str(self.precision_all[min_index])+' '+str(self.recall_all[min_index])
        self.oper_threshold = self.thresholds[min_index]
        self.dice = dice_coefficient_batch(pred, gt, eer_thresh=self.oper_threshold)

        # computed at the operating threshold
        self.Fscore = 2 * (
            (self.recall * self.precision) / (self.recall + self.precision))

        self.evaluated = True

    def print_results(self):
        def collect(*args):
            tup = tuple(args)
            aline = ' '.join(map(str, tup))
            aline += '\n'
            return aline

        res = ''
        if (self.evaluated):
            res += collect('Dice ', np.mean(self.dice, axis=0), '+-', np.std(self.dice, axis=0))
            res += collect('Precision ', self.precision)
            res += collect('Recall ', self.recall)

            res += collect('F score ', self.Fscore)
            res += collect('Operating threshold ', self.oper_threshold)
            print(res)
        return res

    def pr_curve(self):
        # import matplotlib
        # matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.clf()
        lw = 2
        l2d = plt.plot(self.recall_all, self.precision_all, lw=lw, color='navy',
                       label='method1 ')

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        # plt.title('Precision-Recall example: AUC={0:0.2f}'.format(average_precision[0]))
        plt.legend(loc="lower left")

        im = figure2image(l2d[0].figure)
        print('Size of precision-recall image ' + str(np.shape(im)))
        return Image.fromarray(im)


class EvalSegmentationMultiClass:
    def __init__(self, preds, labels, use_threshold=False, mask_ignore=None, background_index=None, eval_class_indices=None):

        """
        Evaluates the segmentation by computing dice coefficient, precision, recall,
        :param preds: NxHxWxC prediction masks  where pixel values are between [0,1] or
        :param labels: NxHxWxC ground truth masks where pixel values are between [0,1]
        :param use_threshold if False use argmax to get class label otherwise use threshold
        :param mask_ignore, NxHxW boolean matrix where true value denotes the pixel should be ignored
        :param eval_class_indices, indices of classes to be evaluated, if None, all indices will be evaluated
        """

        preds = preds.copy()
        labels = labels.copy()

        print('Unique values of labels', np.unique(labels))

        class_preds = np.argmax(preds, axis=3)
        class_labels = np.argmax(labels, axis=3)

        if (mask_ignore is not None):
            mask_ignore= mask_ignore.copy()
            print('Using mask to ignore certain ppreds')
            if (background_index is None):
                bg_index = labels.shape[3] - 1
            ind = np.where(mask_ignore)
            class_preds[ind] = bg_index
            #class_labels[ind] = bg_index

            preds[ind]=0
            last_channel=preds[:,:,:,-1]
            last_channel[ind] =1
            preds[:,:,:,-1]=last_channel

        self.evals = []

        if(eval_class_indices is None):
            eval_class_indices = range(preds.shape[3])

        for i in eval_class_indices: #range(preds.shape[3]):

            if (not use_threshold):
                mask_pred = np.uint8(class_preds == i)
                mask_pred = np.expand_dims(mask_pred, -1)
            else:
                mask_pred = preds[:, :, :, [i]]

            mask_labels = np.expand_dims(np.uint8(class_labels == i), -1)

            self.evals.append(EvalSegmentation(mask_pred, mask_labels))
            # self.evals.append(EvalSegmentation(mask_pred, labels[:, :, :, [i]]))

    def print_results(self):

        res = ''
        for c, ev in enumerate(self.evals):
            res+='Class '+str(c)+'\n'
            res += ev.print_results()

        return res

    def pr_curve(self, colors=None):

        # import matplotlib
        # matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.clf()
        lw = 2
        N_class = len(self.evals)
        # colors = ['red', 'navy', 'yellow', 'orange','blue','green']
        if (colors is None):
            colors = [colorsys.hsv_to_rgb(x * 1.0 / N_class, 0.6, 1) for x in range(N_class)]

        cc = 0
        l2d = None
        for ev in self.evals:
            l2d = plt.plot(ev.recall_all, ev.precision_all, lw=lw, color=colors[cc],
                           label='class' + str(cc))
            plt.plot(ev.precision, ev.recall,'*g')
            plt.plot(0.906, 0.906,'Hb')

            cc += 1

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        # plt.title('Precision-Recall example: AUC={0:0.2f}'.format(average_precision[0]))
        plt.legend(loc="lower left")

        im = figure2image(l2d[0].figure)
        print('Size of precision-recall image ' + str(np.shape(im)))
        return Image.fromarray(im)

    def get_dice_coefficients(self):
        dices = []
        for ev in self.evals:
            dices.append(ev.dice)

        return np.asarray(dices)

    def get_mean_dice_coefficients(self):
        """
        Per class dice coefficient
        :return:
        """
        dices = self.get_dice_coefficients()
        dices_s = []  # save mean dice coef and std as tuples
        for dd in dices:
            dices_s.append([np.mean(dd), np.std(dd)])

        dices_all = np.concatenate(dices)
        dices_s.append([np.mean(dices_all), np.std(dices_all)])
        return dices_s
