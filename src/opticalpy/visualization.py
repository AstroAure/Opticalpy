import autograd.numpy as np
import astropy.units as u
import matplotlib.pyplot as plt

from . import ray
from .elems import mirror

class Camera(mirror.FlatMirror):
    def __init__(self,
                 aperture:u.Quantity|float,
                 scene=None,
                 group=None, 
                 origin:np.ndarray|list[float]=[0,0], 
                 rotation:float|u.Quantity=0*u.deg,
                 label:str = "Camera"):
        self.hits : list[float] = []
        self.colors : list[list[float]] = []
        self.aperture : float = aperture.to(u.mm).value if type(aperture)==u.Quantity else aperture
        super().__init__(aperture, scene, group, origin, rotation, label)

    def interaction(self, angle:u.Quantity, ray:ray.Ray, t_opt:float) -> u.Quantity:
        self.hits.append(t_opt)
        self.colors.append(ray.color)
        return None
    
    def plot(self, ax=None):
        ax = plt.subplots(figsize=(8,2))[1] if ax is None else ax
        ax.scatter(self.hits, [0]*len(self.hits), c=self.colors, s=1)
        ax.set_xlim(self.extent[0], self.extent[1])
        plt.show()