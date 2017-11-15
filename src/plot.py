import torch
import visdom

viz = visdom.Visdom()


class Plot:
    def __init__(self, name):
        self.name = name
        self.plot = viz.line(
            X=torch.zeros((1,)).cpu(),
            Y=torch.zeros((1,)).cpu(),
            opts=dict(
                xlabel="Iteration",
                ylabel="Loss",
                title=name,
            )
        )

    def update(self, step, loss):
        viz.line(
            X=torch.ones((1, 1)).cpu() * step,
            Y=loss.unsqueeze(0).cpu().data.numpy(),
            win=self.plot,
            update='append'
        )
