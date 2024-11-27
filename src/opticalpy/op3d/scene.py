import matplotlib.pyplot as plt
from . import ray, optic

class Scene():
    def __init__(self, xlim:list[float]=None, ylim:list[float]=None, zlim:list[float]=None,
                 lifetime:float=1000, step:float=1, plot_res:int=20) -> None:
        self.rays : list[ray.Ray] = []
        self.optics : list[optic.Optic] = []
        self.xlim, self.ylim, self.zlim = xlim, ylim, zlim
        self.lifetime : float = lifetime
        self.step : float = step
        self.plot_res : int = plot_res

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
            elem.hitbox = optic.calc_hitbox(elem.eq, elem.eq_bounds, size=elem.size, expand=self.step)
            elem.plot_res = self.plot_res
            elem.update_transforms()

    def plot(self, ax=None, show_hitbox=False, show=True) -> None:
        ax = plt.subplots(figsize=(8,6), subplot_kw={'projection':'3d'})[1] if ax is None else ax
        for r in self.rays:
            r.__plot__(ax, self.lifetime)
        for o in self.optics:
            o.__plot__(ax)
            if show_hitbox: o.plot_hitbox(ax)
        if self.xlim is not None: ax.set_xlim(self.xlim[0],self.xlim[1])
        if self.ylim is not None: ax.set_ylim(self.ylim[0],self.ylim[1])
        if self.zlim is not None: ax.set_zlim(self.zlim[0],self.zlim[1])
        ax.set_aspect('equal', 'datalim')
        ax.set_xlabel('X [mm]')
        ax.set_ylabel('Y [mm]')
        ax.set_zlabel('Z [mm]')
        if show: plt.show()