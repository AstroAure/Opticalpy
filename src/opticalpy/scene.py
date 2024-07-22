import matplotlib.pyplot as plt
from . import ray, optic, optgroup

class Scene():
    def __init__(self, xlim:list[float]=None, ylim:list[float]=None,
                 lifetime:float=1000, step:float=1):
        self.rays : list[ray.Ray] = []
        self.optics : list[optic.Optic] = []
        self.xlim, self.ylim = xlim, ylim
        self.lifetime : float = lifetime
        self.step :float = step

    def __str__(self) -> str:
        rays = ", ".join([elem.label for elem in self.rays])
        optics = ", ".join([elem.label for elem in self.optics])
        return (f"Rays   : {rays}\nOptics : {optics}")

    def append(self, elem) -> None:
        elem.scene = self
        if isinstance(elem, ray.Ray):
            self.rays.append(elem)
            elem.step = self.step
        if isinstance(elem, optic.Optic):
            self.optics.append(elem)
            elem.hitbox = elem.calc_hitbox(self.step)
        if isinstance(elem, optgroup.OpticalGroup):
            for e in elem.elements:
                self.append(e)

    def plot(self, ax=None, show_hitbox=False, show=True) -> None:
        ax = plt.subplots(figsize=(8,6))[1] if ax is None else ax
        for r in self.rays:
            r.__plot__(ax, self.lifetime)
        for o in self.optics:
            o.__plot__(ax)
            if show_hitbox: o.plot_hitbox(ax)
        ax.set_aspect('equal')
        if self.xlim is not None: ax.set_xlim(self.xlim[0],self.xlim[1])
        if self.ylim is not None: ax.set_ylim(self.ylim[0],self.ylim[1])
        if show: plt.show()