import numpy as np
from PIL import Image
import matplotlib

matplotlib.use('Agg')


class ImageVizMPL:
    def __init__(self):


        import matplotlib.pyplot as plt
        self.plt = plt
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        self.FigureCanvasAgg = FigureCanvasAgg

    def get_heatmap(self, im):
        """
        Generates a heatmap image from a  probability map using matplotlib
        :param im: HxW
        :param load_mpl:
        :return:
        """

        plt = self.plt
        plt.clf()

        img = plt.imshow(im, aspect='auto')
        img.set_cmap('jet_r')  # reverse jet colormap
        plt.axis('off')

        img.axes.get_xaxis().set_visible(False)
        img.axes.get_yaxis().set_visible(False)
        plt.gcf().subplots_adjust(bottom=0)
        plt.gcf().subplots_adjust(top=1)
        plt.gcf().subplots_adjust(right=1)
        plt.gcf().subplots_adjust(left=0)

        fim = self.figure2image(plt.gcf())
        # print 'obtained from fig2img'
        fim = Image.fromarray(fim)
        fim = fim.resize((im.shape[1], im.shape[0]), Image.BILINEAR)
        return np.asarray(fim)

    def figure2image(self, fig):
        """
         Returns the matplotlib figure as an image
        :param fig:  matplotlib figure object
        :load_mpl: if True loads matplotlib library. Set it it False if already loaded
        :return: image
        Examples:
        ________
        >>> import matplotlib.pyplot as plt
        >>> myplot = plt.plot(x,y)
        >>> im = figure2image(plt.figure())

        """
        canvas = self.FigureCanvasAgg(fig)
        canvas.draw()  # draw the canvas, cache the renderer

        width, height = fig.get_size_inches() * fig.get_dpi()

        image = np.fromstring(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)

        return image

    def plot_curves(self, curves, legends, xlabel, ylabel, colors=None, title=None):

        plt = self.plt
        plt.clf()
        plt.axis('on')
        lw = 2
        if (colors is None):
            colors = ['red', 'navy', 'yellow', 'orange', 'blue', 'green']
        cc = 0
        l2d = None
        for cur, leg in zip(curves, legends):
            l2d = plt.plot(cur, lw=lw, color=colors[cc], label=leg)
            cc += 1

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        # plt.ylim([0.0, 1.05])
        # plt.xlim([0.0, 1.0])
        if (title):
            plt.title(title)

        plt.legend(loc="upper right")

        im = self.figure2image(l2d[0].figure)
        return im
