from __future__ import print_function

import os
from random import randint, uniform

import matplotlib.pyplot as plt
import nibabel as nib
import torch
from natsort import natsorted

plt.rcParams["image.cmap"] = "gray_r"

from datetime import datetime
from time import time

import plotly.graph_objects as go
import plotly.graph_objs as graphs
import plotly.offline as ply
from plotly.subplots import make_subplots


def listdir_nohidden(path):
    """
    List files in directory ignoring hidden files (starting with a point)

    Args:
        path (str): Path to directory

    Yields:
        generator object
    """

    for file in natsorted(os.listdir(path)):
        if not file.startswith("."):
            yield file


def load_nii_image_as_npy(file):

    img = nib.load(file)
    img = img.get_fdata()

    return img


def plot_loss(losses, filename, show=True):
    """Plot loss over all epochs."""

    x = list(losses.keys())
    y = [i for i in losses.values()]

    trace_train = graphs.Scatter(
        x=x,
        y=y,
        name="Training",
        mode="lines+markers",
        line=dict(width=4),
        marker=dict(symbol="circle", size=10),
    )

    title = os.path.basename(filename).split(".")[0]
    layout = graphs.Layout(
        title=title, xaxis={"title": "Epoch"}, yaxis={"title": "Loss"}
    )

    fig = graphs.Figure(data=[trace_train], layout=layout)
    ply.plot(fig, show_link=True, filename=filename, auto_open=show)


def plot_disc_gen_loss(disc_losses, gen_losses, filename):

    epochs = [i + 1 for i, _ in enumerate(disc_losses)]

    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add traces
    fig.add_trace(
        go.Scatter(x=epochs, y=disc_losses, name="Discriminator (D)"),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(x=epochs, y=gen_losses, name="Generator GAN (G_GAN)"),
        secondary_y=True,
    )

    # Add figure title
    fig.update_layout(title_text="<b>Training history<b>")

    # Set x-axis title
    fig.update_xaxes(title_text="<b>Epoch<b>")

    # Set y-axes titles
    fig.update_yaxes(title_text="<b>Loss D</b>", secondary_y=False)
    fig.update_yaxes(title_text="<b>Loss G_GAN</b>", secondary_y=True)

    # fig.show()
    ply.plot(fig, show_link=True, filename=filename, auto_open=False)


class DecayLR:
    def __init__(self, epochs, offset, decay_epochs):
        epoch_flag = epochs - decay_epochs
        assert epoch_flag > 0, "Decay must start before the training session ends!"
        self.epochs = epochs
        self.offset = offset
        self.decay_epochs = decay_epochs

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_epochs) / (
            self.epochs - self.decay_epochs
        )


class ReplayBuffer:
    def __init__(self, max_size=50):
        assert (
            max_size > 0
        ), "Empty buffer or trying to create a black hole. Be careful."
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if uniform(0, 1) > 0.5:
                    i = randint(0, self.max_size - 1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return torch.cat(to_return)


class IndexTracker(object):
    """
    Image Viewer for 3D Numpy arrays such as volumetric medical images.
    Ref: https://matplotlib.org/gallery/animation/image_slices_viewer.html
    """

    def __init__(self, ax, X, view, vmin=None, vmax=None):
        self.ax = ax
        ax.set_title("use scroll wheel to navigate images")

        self.X = X
        rows, cols, self.slices = X.shape

        self.view = view
        self.ind = 0

        # axial
        if self.view == "axial":
            self.im = ax.imshow(self.X[:, :, self.ind], vmin=vmin, vmax=vmax)

        # sagittal
        if self.view == "sagittal":
            self.im = ax.imshow(self.X[self.ind, :, :], vmin=vmin, vmax=vmax)

        # coronal
        if self.view == "coronal":
            self.im = ax.imshow(self.X[:, self.ind, :], vmin=vmin, vmax=vmax)

        self.update()

    def onscroll(self, event):
        print("%s %s" % (event.button, event.step))
        if event.button == "up":
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        self.update()

    def update(self):
        # axial
        if self.view == "axial":
            self.im.set_data(self.X[:, :, self.ind])

        # sagittal
        if self.view == "sagittal":
            self.im.set_data(self.X[self.ind, :, :])

        # coronal
        if self.view == "coronal":
            self.im.set_data(self.X[:, self.ind, :])

        self.ax.set_ylabel(f"slice {self.ind+1}")
        self.im.axes.figure.canvas.draw()


def plot_3d(img, vmin, vmax, view="axial", show=False):
    print(f"Size: {img.shape}")
    fig, ax = plt.subplots(1, 1)
    tracker = IndexTracker(ax, img, view=view, vmin=vmin, vmax=vmax)
    fig.canvas.mpl_connect("scroll_event", tracker.onscroll)
    if show:
        plt.show()


class GenericLogger(object):
    """
    Ref: https://stackoverflow.com/questions/11325019/how-to-output-to-the-console-and-file
    """

    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()  # If you want the output to be visible immediately

    def flush(self):
        for f in self.files:
            f.flush()


def print_with_timestamp(*args):
    timestamp = time()
    dt_object = datetime.fromtimestamp(timestamp)
    args = ("%s:" % dt_object, *args)
    print(*args)


def test():
    import torch
    import torch.optim as optim

    optimizer = optim.SGD([torch.rand((2, 2), requires_grad=True)], lr=1)
    # scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=0, total_iters=1000)

    # c = 0
    # for epoch in range(0, 1000):
    #     c += 1
    #     scheduler.step()
    #     print(f'Epoch-{epoch} lr: {optimizer.param_groups[0]["lr"]}')
    # print(c)

    lambda_lr = DecayLR(epochs=1000, offset=0, decay_epochs=500).step
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda_lr)

    c = 0
    for epoch in range(0, 1000):
        c += 1
        scheduler.step()
        print(f'Epoch-{epoch} lr: {optimizer.param_groups[0]["lr"]}')
    print(c)


if __name__ == "__main__":
    test()
