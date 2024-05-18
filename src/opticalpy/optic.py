import autograd.numpy as np
import astropy.units as u
from autograd import grad

from . import ray

class Optic():
    def __init__(self, 
                 equations:list, 
                 extent:list[float],
                 scene:None=None,
                 group=None, 
                 origin:np.ndarray|list[float]=[0,0], 
                 rotation:float|u.Quantity=0*u.deg,
                 label:str|None = None,
                 color='k'):
        self.raw_equation_x = equations[0]
        self.raw_equation_y = equations[1]
        self.extent : list[float] = extent
        self.origin : np.ndarray = origin if type(origin)==np.ndarray else np.array(origin)
        self.rotation = rotation if type(rotation)==u.Quantity else rotation*u.deg
        self.sin, self.cos = np.sin(self.rotation).value, np.cos(self.rotation).value
        self.equation_x, self.equation_y = self.calc_equation()
        self.label : str|None = label
        self.color = color
        self.scene = scene
        self.group = group
        if (self.scene is not None) & (self.group is None): self.scene.append(self)
        if self.group is not None: self.group.append(self)
        if self.scene is not None: self.hitbox : list[float] = self.calc_hitbox(self.scene.step)

    def calc_equation(self):
        equation_x = lambda t : (self.cos*self.raw_equation_x(t) - self.sin*self.raw_equation_y(t) + self.origin[0])
        equation_y = lambda t : (self.sin*self.raw_equation_x(t) + self.cos*self.raw_equation_y(t) + self.origin[1])
        return equation_x, equation_y

    def calc_hitbox(self, expand:float=1.0) -> list[float]:
        t = np.linspace(self.extent[0], self.extent[1], 100)
        x, y = self.equation_x(t), self.equation_y(t)
        left, right = np.min(x)-expand, np.max(x)+expand
        bottom, top = np.min(y)-expand, np.max(y)+expand 
        return left, right, bottom, top
    
    def plot_hitbox(self, ax) -> None:
        l,r,b,t = self.hitbox
        ax.plot([l,r,r,l,l], [b,b,t,t,b], color='r', ls=':', alpha=0.5)

    def __plot__(self, ax) -> None:
        t = np.linspace(self.extent[0], self.extent[1], 100)
        x, y = self.equation_x(t), self.equation_y(t)
        ax.plot(x, y, color=self.color)

    def normal_angle(self, t:float) -> u.Quantity:
        equation_dx = grad(self.equation_x)
        equation_dy = grad(self.equation_y)
        normal = np.array([-equation_dy(t), equation_dx(t)])
        normal /= np.linalg.norm(normal)
        normal_angle = np.arctan2(normal[1], normal[0])*u.rad
        return normal_angle

    def interaction(self, angle:u.Quantity, ray:ray.Ray, t_opt:float) -> u.Quantity:
        pass

    def update_pos(self, new_pos:list[float]=None, new_rot:u.Quantity|float=None):
        new_pos = self.origin if new_pos is None else new_pos
        new_rot = self.rotation if new_rot is None else new_rot
        new_pos = new_pos if type(new_pos)==np.ndarray else np.array(new_pos)
        new_rot = new_rot if type(new_rot)==u.Quantity else new_rot*u.deg
        self.origin = new_pos
        self.rotation = new_rot
        self.sin, self.cos = np.sin(self.rotation).value, np.cos(self.rotation).value
        self.equation_x, self.equation_y = self.calc_equation()
        if self.scene is not None: self.hitbox = self.calc_hitbox(self.scene.step)