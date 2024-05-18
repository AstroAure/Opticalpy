import autograd.numpy as np
import astropy.units as u
from scipy import optimize

from . import utils

class Ray():
    def __init__(self,
                 wavelength:float|u.Quantity=450*u.nm,
                 scene=None,
                 group=None,
                 origin:np.ndarray|list[float]=[0,0],
                 rotation:float|u.Quantity=0*u.deg,
                 label:str|None="Ray"):
        
        self.wavelength : u.Quantity = wavelength if type(wavelength)==u.Quantity else wavelength*u.nm
        self.n_in : float = 1.0
        self.origin : np.ndarray = origin if type(origin)==np.ndarray else np.array(origin)
        self.rotation: u.Quantity = rotation if type(rotation)==u.Quantity else rotation*u.deg
        self.color : list[float] = utils.wavelength2RGB(self.wavelength)
        self.label : str|None = label
        self.scene = scene
        self.group = group
        if (self.scene is not None) & (self.group is None): self.scene.append(self)
        if self.group is not None: self.group.append(self)
        if self.scene is not None: self.step : float = self.scene.step

    def find_next_collision(self, origin:np.ndarray, rotation:u.Quantity, lifetime:float):
        # Calculate points along ray
        s = np.arange(0, lifetime, self.step)
        cos, sin = np.cos(rotation).value, np.sin(rotation).value
        eq_ray_x = lambda s : cos*s + origin[0]
        eq_ray_y = lambda s : sin*s + origin[1]
        x, y = eq_ray_x(s), eq_ray_y(s)
        
        # For each optical element, test collision with hitbox...
        opt_collisions = []
        s_collisions = [] # Parameter on ray
        t_collisions = [] # Parameter on optic
        for optic in self.scene.optics:
            l,r,b,t = optic.hitbox
            collision = np.any((x>l)&(x<r)&(y>b)&(y<t))
            if collision: 
                # ... and with actuel element
                # print(f"Hitbox collision between {self.label} and {optic.label}")
                func = lambda s: (eq_ray_x(s[0])-optic.equation_x(s[1]),eq_ray_y(s[0])-optic.equation_y(s[1]))
                st_collision = optimize.fsolve(func, [0,0])
                t_min = min(optic.extent[0], optic.extent[1])
                t_max = max(optic.extent[0], optic.extent[1])
                if (st_collision[1]>t_min)&(st_collision[1]<t_max)&(st_collision[0]>self.step):
                    # print(f"Collision between {self.label} and {optic.label} : {st_collision[0]}")
                    s_collisions.append(st_collision[0])
                    t_collisions.append(st_collision[1])
                    opt_collisions.append(optic)

        # No collision case
        if len(s_collisions)==0:
            return [x[-1],y[-1]], rotation, 0
        
        # Collision case
        idx = np.argmin(s_collisions)
        optic_coll = opt_collisions[idx]
        pos_coll = [eq_ray_x(s_collisions[idx]), eq_ray_y(s_collisions[idx])]
        # print(f"Collision between {self.label} and {optic_coll.label} : {pos_coll[0]:.2f}, {pos_coll[1]:.2f}")
        ray = -np.array([np.cos(rotation), np.sin(rotation)])
        ray_angle = np.arctan2(ray[1], ray[0])*u.rad
        normal_angle = optic_coll.normal_angle(t_collisions[idx])
        angle = utils.dist_angle(normal_angle, ray_angle)
        if abs(angle)>90*u.deg: 
            normal_angle += 180*u.deg
            angle = utils.dist_angle(normal_angle, ray_angle)
        out_angle = optic_coll.interaction(angle, self, t_collisions[idx])
        if out_angle is None:
            return pos_coll, ray_angle+180*u.deg, 0
        rotation_coll = out_angle + normal_angle
        rotation_coll = rotation_coll%(360*u.deg)
        return pos_coll, rotation_coll, lifetime-s_collisions[idx]
    
    def propagate(self, lifetime:float=1000):
        points = [[self.origin[0],self.origin[-1]]]
        rotation = self.rotation
        # Finds next collision as long as lifetime is not 0
        while lifetime > 0:
            point, rotation, lifetime = self.find_next_collision(points[-1], rotation, lifetime)
            points.append(point)
        return np.array(points)

    def __plot__(self, ax, lifetime:float=1000) -> None:
        pos = self.propagate(lifetime)
        ax.plot(pos[:,0], pos[:,1], color=self.color, lw=0.5)
        ax.scatter(self.origin[0], self.origin[1], color=[self.color])

    def update_pos(self, new_pos:list[float]=None, new_rot:u.Quantity|float=None):
        new_pos = self.origin if new_pos is None else new_pos
        new_rot = self.rotation if new_rot is None else new_rot
        new_pos = new_pos if type(new_pos)==np.ndarray else np.array(new_pos)
        new_rot = new_rot if type(new_rot)==u.Quantity else new_rot*u.deg
        self.origin = new_pos
        self.rotation = new_rot