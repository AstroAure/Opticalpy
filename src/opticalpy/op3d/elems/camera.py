import autograd.numpy as np
import astropy.units as u
import matplotlib.pyplot as plt

from .. import ray
from . import mirror

class Camera(mirror.FlatMirror):
    def __init__(self,
                 width:u.Quantity|float,
                 height:u.Quantity|float,
                 label:str = "Camera",
                 **kwargs):
        self.width : float = width.to_value(u.mm) if type(width)==u.Quantity else width
        self.height : float = height.to_value(u.mm) if type(height)==u.Quantity else height
        super().__init__(diameter=np.maximum(width,height), label=label, **kwargs)
        self.eq_bounds = (lambda x,y,z: np.maximum(abs(x)-0.5*self.width,abs(y)-0.5*self.height))
        self.update_transforms()
        self.hits : list[float] = []
        self.colors : list[list[float]] = []

    def interaction(self, dir:np.ndarray, normal:np.ndarray, r:ray.Ray, pos_coll:np.ndarray) -> u.Quantity:
        rel_hit = pos_coll - self.origin
        self.hits.append(rel_hit)
        self.colors.append(r.color)
        return None
    
    def plot(self, ax=None):
        ax = plt.subplots(figsize=(8,2))[1] if ax is None else ax
        ax.scatter(np.tensordot(self.hits, self.rot_matrix@[1,0,0], axes=1), np.tensordot(self.hits, self.rot_matrix@[0,1,0], axes=1), c=self.colors, s=1)
        ax.set_xlim(-0.5*self.width, 0.5*self.width)
        ax.set_ylim(-0.5*self.height, 0.5*self.height)
        ax.set_aspect('equal', 'box')
        ax.set_title(self.label)
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")
        plt.show()