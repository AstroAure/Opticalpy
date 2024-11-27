import autograd.numpy as np
import astropy.units as u
from scipy import optimize

from opticalpy import utils

class Ray():
    def __init__(self,
                 wavelength:float|u.Quantity=450*u.nm,
                 order:int=1,
                 scene=None,
                 origin:np.ndarray|list[float]=[0,0,0],
                 direction:np.ndarray|list[float]=[1,0,0],
                 label:str|None="Ray",
                 step:float=0.1,
                 plot_alpha:float=1,) -> None:
        
        self.wavelength : u.Quantity = wavelength if type(wavelength)==u.Quantity else wavelength*u.nm
        self.order : int = order
        self.origin : np.ndarray = np.array(origin).astype(float)
        self.direction: np.ndarray = np.array(direction).astype(float)
        self.direction /= np.linalg.norm(self.direction)
        self.color : list[float] = utils.wavelength2RGB(self.wavelength)
        self.label : str|None = label
        self.step : float = step
        self.plot_alpha : float = plot_alpha
        self.scene = scene
        if (self.scene is not None): self.scene.append(self)

    def find_next_collision(self, origin:np.ndarray, direction:np.ndarray, lifetime:float, debug:bool=False):
        # Calculate points along ray
        s = np.arange(0, lifetime, self.step)
        eq_ray = lambda s : np.tensordot(s, direction, axes=0) + origin
        pos = eq_ray(s)
        x,y,z = pos.T
        # For each optical element, test collision with hitbox...
        opt_collisions = []
        s_collisions = [] # Parameter on ray
        t_collisions = [] # Position on optic
        for opt in self.scene.optics:
            if debug: print(f"Testing collision between {self.label} and {opt.label}")
            x0,x1 = opt.trans_hitbox['x'][0], opt.trans_hitbox['x'][1]
            y0,y1 = opt.trans_hitbox['y'][0], opt.trans_hitbox['y'][1]
            z0,z1 = opt.trans_hitbox['z'][0], opt.trans_hitbox['z'][1]
            collision = np.any((x>x0)&(x<x1)&(y>y0)&(y<y1)&(z>z0)&(z<z1))
            if debug==3: print(opt.trans_hitbox)
            if debug==3: print(np.any(x>x0), np.any(x<x1), np.any(y>y0), np.any(y<y1), np.any(z>z0), np.any(z<z1))
            if collision: 
                # ... and with actuel element
                if debug: print(f"Hitbox collision between {self.label} and {opt.label}")
                def func(s):
                    collision = eq_ray(s[0])-np.array([s[1],s[2],s[3]])
                    on_surf = opt.trans_eq(s[1],s[2],s[3])
                    return np.array([*list(collision), on_surf])
                # Calculates potential multiple collisions
                colls = []
                for x_ in opt.trans_hitbox['x']:
                    for y_ in opt.trans_hitbox['y']:
                        for z_ in opt.trans_hitbox['z']:
                            st_collision = optimize.root(func, x0=[0,x_,y_,z_], tol=1e-6).x
                            colls.append(st_collision)
                colls = np.array(colls)
                if debug==2: print(colls)
                # Ensure that collision is not at last collision
                roots_idx = np.unique(colls.round(5), axis=0, return_index=True)[1]
                roots = colls[roots_idx]
                if debug==2: print(roots)
                wrong_coll = np.logical_or(np.isclose(roots[:,0], 0, rtol=1e-5),roots[:,0]<0)
                if np.where(~wrong_coll)[0].size == 0:
                    continue
                else:
                    # Find first collision in positive direction
                    roots = np.delete(roots, np.where(wrong_coll)[0], axis=0)
                    st_collision = roots[np.argmin(roots[:,0])]
                    if debug==2: print(st_collision)
                # Verify if collision is on surface and in bounds
                if np.max(np.abs(func(st_collision)) < 1e-6) & (opt.trans_eq_bounds(*st_collision[1:]) <= 0):
                    if debug: print(f"Collision between {self.label} and {opt.label} : {st_collision[0]}")
                    s_collisions.append(st_collision[0])
                    t_collisions.append(st_collision[1:])
                    opt_collisions.append(opt)
        # No collision case
        if len(s_collisions)==0:
            return pos[-1], direction, 0
        # Collision case
        idx = np.argmin(s_collisions)
        optic_coll = opt_collisions[idx]
        pos_coll = t_collisions[idx]
        if debug: print(f"Collision between {self.label} and {optic_coll.label} : {pos_coll[0]:.2f}, {pos_coll[1]:.2f}, {pos_coll[2]:.2f}")
        normal = optic_coll.trans_normal(pos_coll)
        normal = normal/np.linalg.norm(normal)
        out_dir = optic_coll.interaction(direction, normal, self, pos_coll)
        if out_dir is None:
            return pos_coll, direction, 0
        return pos_coll, out_dir, lifetime-s_collisions[idx]
    
    def propagate(self, lifetime:float=1000, debug:bool=False) -> np.ndarray:
        points = [self.origin.copy()]
        direction = self.direction.copy()
        # Finds next collision as long as lifetime is not 0
        while lifetime > 0:
            point, direction, lifetime = self.find_next_collision(points[-1], direction, lifetime)
            if debug: print(f"New point : {point} | New direction : {direction} | Lifetime : {lifetime}")
            points.append(point)
        return np.array(points)

    def __plot__(self, ax, lifetime:float=1000, **kwargs) -> None:
        pos = self.propagate(lifetime)
        ax.plot(pos[:,0], pos[:,1], pos[:,2], color=self.color, lw=1, alpha=self.plot_alpha, **kwargs)
        ax.scatter(*self.origin, color=[self.color], label=self.label)

    def set_origin(self, new_origin:np.ndarray|list[float]) -> None:
        new_origin = np.array(new_origin)
        self.origin = new_origin

    def set_direction(self, new_direction:np.ndarray|list[float]) -> None:
        new_direction = np.array(new_direction)
        new_direction /= np.linalg.norm(new_direction)
        self.direction = new_direction

    # def update_pos(self, new_pos:list[float]=None, new_rot:u.Quantity|float=None):
    #     new_pos = self.origin if new_pos is None else new_pos
    #     new_rot = self.rotation if new_rot is None else new_rot
    #     new_pos = new_pos if type(new_pos)==np.ndarray else np.array(new_pos)
    #     new_rot = new_rot if type(new_rot)==u.Quantity else new_rot*u.deg
    #     self.origin = new_pos
    #     self.rotation = new_rot