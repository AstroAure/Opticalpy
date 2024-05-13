import autograd.numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from scipy import optimize
from autograd import grad

### UTILS ###

def wavelength2RGB(wavelength:u.Quantity) -> list[float]:
    # Human like RGB color filters
    w = (wavelength.to(u.nm)).value
    def piecewise_gaussian(x, mu, tau1, tau2):
        y = np.empty_like(x)
        y[x<mu] = np.exp(-tau1**2*(x-mu)**2/2)[x<mu]
        y[x>=mu] = np.exp(-tau2**2*(x-mu)**2/2)[x>=mu]
        return y
    x = 1.056*piecewise_gaussian(w, 599.8,0.0264,0.0323) + \
        0.362*piecewise_gaussian(w, 442.0, 0.0624, 0.0374) - \
        0.065*piecewise_gaussian(w, 501.1, 0.0490, 0.0382)
    y = 0.821*piecewise_gaussian(w, 568.8,0.0213,0.0247) + \
        0.286*piecewise_gaussian(w, 530.9, 0.0613, 0.0322)
    z = 1.217*piecewise_gaussian(w, 437.0,0.0845,0.0278) + \
        0.681*piecewise_gaussian(w, 459.0, 0.0385, 0.0725)
    XYZ2RGB = np.array([[2.36461385, -0.89654057, -0.46807328],
                        [-0.51516621, 1.4264081,   0.0887581],
                        [0.0052037,  -0.01440816, 1.00920446]])
    rgb = XYZ2RGB@[x,y,z]
    rgb = np.clip(rgb, 0, 1)
    return rgb

def dist_angle(ang1:u.Quantity, ang2:u.Quantity) -> u.Quantity:
    # Signed angle from ang1 to ang2, result in (-180deg,180deg]
    return np.arctan2(np.sin(ang2-ang1), np.cos(ang2-ang1))

### SCENE ###

class Scene():
    def __init__(self, xlim:list[float]=None, ylim:list[float]=None,
                 lifetime:float=1000, step:float=1):
        self.rays : list[Ray] = []
        self.optics : list[Optic] = []
        self.xlim, self.ylim = xlim, ylim
        self.lifetime : float = lifetime
        self.step :float = step

    def __str__(self) -> str:
        rays = ", ".join([elem.label for elem in self.rays])
        optics = ", ".join([elem.label for elem in self.optics])
        return (f"Rays   : {rays}\nOptics : {optics}")

    def append(self, elem) -> None:
        elem.scene = self
        if isinstance(elem, Ray):
            self.rays.append(elem)
            elem.step = self.step
        if isinstance(elem, Optic):
            self.optics.append(elem)
            elem.hitbox = elem.calc_hitbox(self.step)
        if isinstance(elem, OpticalGroup):
            for e in elem.elements:
                self.append(e)

    def plot(self, ax=None, show_hitbox=False) -> None:
        ax = plt.subplots(figsize=(8,6))[1] if ax is None else ax
        for ray in self.rays:
            ray.__plot__(ax, self.lifetime)
        for optic in self.optics:
            optic.__plot__(ax)
            if show_hitbox: optic.plot_hitbox(ax)
        ax.set_aspect('equal')
        if self.xlim is not None: ax.set_xlim(self.xlim[0],self.xlim[1])
        if self.ylim is not None: ax.set_ylim(self.ylim[0],self.ylim[1])
        plt.show()

### RAYS ###

class Ray():
    def __init__(self,
                 wavelength:float|u.Quantity=450*u.nm,
                 scene:Scene|None=None,
                 group=None,
                 origin:np.ndarray|list[float]=[0,0],
                 rotation:float|u.Quantity=0*u.deg,
                 label:str|None="Ray"):
        
        self.wavelength : u.Quantity = wavelength if type(wavelength)==u.Quantity else wavelength*u.nm
        self.n_in : float = 1.0
        self.origin : np.ndarray = origin if type(origin)==np.ndarray else np.array(origin)
        self.rotation: u.Quantity = rotation if type(rotation)==u.Quantity else rotation*u.deg
        self.color : list[float] = wavelength2RGB(self.wavelength)
        self.label : str|None = label
        self.scene : Scene = scene
        self.group : OpticalGroup = group
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
        angle = dist_angle(normal_angle, ray_angle)
        if abs(angle)>90*u.deg: 
            normal_angle += 180*u.deg
            angle = dist_angle(normal_angle, ray_angle)
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

def WhiteRay(wavelengths:list[u.Quantity|float]|np.ndarray,
             scene:Scene|None=None,
             group=None,
             origin:list[float]|np.ndarray=[0,0], 
             rotation:float|u.Quantity=0.0*u.deg, 
             label:str|None="White") -> list[Ray]:
    rays = []
    for i, wave in enumerate(wavelengths):
        ray = Ray(wavelength=wave, scene=scene, group=group, origin=origin, rotation=rotation, label=f"{label} {i}")
        rays.append(ray)
    return rays

def CollimatedBeam(aperture:float,
                   wavelength:float|u.Quantity|list[u.Quantity|float]|np.ndarray=450*u.nm,
                   scene:Scene|None=None,
                   group=None,
                   origin:list[float]|np.ndarray=[0,0], 
                   rotation:float|u.Quantity=0.0*u.deg, 
                   N_sources:int=20, 
                   hole:float=0.0,
                   label:str|None="Collimated"):
    rotation = rotation.to(u.deg).value if type(rotation)==u.Quantity else rotation
    beam = OpticalGroup(scene=scene, group=group, origin=origin, rotation=rotation, label=label)
    for i, ang in enumerate(np.linspace(-0.5*aperture,0.5*aperture,N_sources)):
        if (hole==0) | (abs(ang)>0.5*hole):
            if (type(wavelength)==int) or (type(wavelength)==float) or (type(wavelength)==u.Quantity):
                ray = Ray(wavelength=wavelength, group=beam, origin=[0,ang], rotation=0, label=i)
            else:
                ray = WhiteRay(wavelengths=wavelength, group=beam, origin=[0,ang], rotation=0, label=i)
    return beam

def DivergingBeam(angle:float|u.Quantity,
                  wavelength:float|u.Quantity|list[u.Quantity|float]|np.ndarray=450*u.nm, 
                  scene:Scene|None=None,
                  group=None,
                  origin:list[float]|np.ndarray=[0,0], 
                  rotation:float|u.Quantity=0.0*u.deg, 
                  N_sources:int=20, 
                  hole:float|u.Quantity=0.0*u.deg,
                  label:str|None="Diverging") -> list[Ray]:
    rotation = rotation.to(u.deg).value if type(rotation)==u.Quantity else rotation
    angle = angle.to(u.deg).value if type(angle)==u.Quantity else angle
    hole = hole.to(u.deg).value if type(hole)==u.Quantity else hole
    beam = OpticalGroup(scene=scene, group=group, origin=origin, rotation=rotation, label=label)
    for i, ang in enumerate(np.linspace(-0.5*angle,0.5*angle,N_sources)):
        if (hole==0) | (abs(ang)>0.5*hole):
            if (type(wavelength)==int) or (type(wavelength)==float) or (type(wavelength)==u.Quantity):
                ray = Ray(wavelength=wavelength, group=beam, origin=[0,0], rotation=ang, label=i)
            else:
                ray = WhiteRay(wavelengths=wavelength, group=beam, origin=[0,0], rotation=ang, label=i)
    return beam

def ConvergingBeam(focal:float|u.Quantity,
                   F:float,
                   wavelength:float|u.Quantity|list[u.Quantity|float]|np.ndarray=450*u.nm,
                   scene:Scene|None=None,
                   group=None,
                   origin:list[float]|np.ndarray=[0,0], 
                   rotation:float|u.Quantity=0.0*u.deg, 
                   N_sources:int=20, 
                   hole:float|u.Quantity=0.0*u.deg,
                   label:str|None="Converging") -> list[Ray]:
    rotation = rotation.to(u.deg).value if type(rotation)==u.Quantity else rotation
    hole = hole.to(u.deg).value if type(hole)==u.Quantity else hole
    angle = 2*np.rad2deg(np.arctan(0.5/F))
    beam = OpticalGroup(scene=scene, group=group, origin=origin, rotation=rotation, label=label)
    for i, ang in enumerate(np.linspace(-0.5*angle,0.5*angle,N_sources)):
        if (hole==0) | (abs(ang)>0.5*hole):
            offset = focal*np.array([1-np.cos(np.deg2rad(ang)),np.sin(np.deg2rad(ang))])
            if (type(wavelength)==int) or (type(wavelength)==float) or (type(wavelength)==u.Quantity):
                ray = Ray(wavelength=wavelength, group=beam, origin=offset, rotation=-ang, label=i)
            else:
                ray = WhiteRay(wavelengths=wavelength, group=beam, origin=offset, rotation=-ang, label=i)
    return beam

### OPTIC ###

class Optic():
    def __init__(self, 
                 equations:list, 
                 extent:list[float],
                 scene:Scene|None=None,
                 group=None, 
                 origin:np.ndarray|list[float]=[0,0], 
                 rotation:float|u.Quantity=0*u.deg,
                 label:str|None = None):
        self.raw_equation_x = equations[0]
        self.raw_equation_y = equations[1]
        self.extent : list[float] = extent
        self.origin : np.ndarray = origin if type(origin)==np.ndarray else np.array(origin)
        self.rotation = rotation if type(rotation)==u.Quantity else rotation*u.deg
        self.sin, self.cos = np.sin(self.rotation).value, np.cos(self.rotation).value
        self.equation_x, self.equation_y = self.calc_equation()
        self.label : str|None = label
        self.scene : Scene = scene
        self.group : OpticalGroup = group
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
        ax.plot(x, y, color='k')

    def normal_angle(self, t:float) -> u.Quantity:
        equation_dx = grad(self.equation_x)
        equation_dy = grad(self.equation_y)
        normal = np.array([-equation_dy(t), equation_dx(t)])
        normal /= np.linalg.norm(normal)
        normal_angle = np.arctan2(normal[1], normal[0])*u.rad
        return normal_angle

    def interaction(self, angle:u.Quantity, ray:Ray, t_opt:float) -> u.Quantity:
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

### OPTIC.MIRROR ##

class Mirror(Optic):
    def __init__(self,
                 equations:list, 
                 extent:list[float],
                 scene:Scene|None=None,
                 group=None,
                 origin:np.ndarray|list[float]=[0,0], 
                 rotation:float|u.Quantity=0*u.deg,
                 label:str|None = "Mirror"):
        super().__init__(equations, extent, scene, group, origin, rotation, label)

    def interaction(self, angle:u.Quantity, ray:Ray, t_opt:float) -> u.Quantity:
        return -angle
    

class FlatMirror(Mirror):
    def __init__(self,
                 aperture:float|u.Quantity,
                 scene:Scene|None=None,
                 group=None,
                 origin:np.ndarray|list[float]=[0,0], 
                 rotation:float|u.Quantity=0*u.deg,
                 label:str = "Flat mirror"):
        self.aperture : float = aperture.to(u.mm).value if type(aperture)==u.Quantity else aperture
        super().__init__(equations=(lambda t: t, lambda t: 0*t), 
                         extent=[-0.5*self.aperture,0.5*self.aperture], 
                         scene=scene, group=group,origin=origin, rotation=rotation, label=label)
        
def Slit(full_size:float|u.Quantity,
         slit_width:float|u.Quantity,
         scene:Scene|None=None,
         group=None,
         origin:np.ndarray|list[float]=[0,0], 
         rotation:float|u.Quantity=0*u.deg,
         label:str = "Slit"):
    full_size = full_size.to(u.mm).value if type(full_size)==u.Quantity else full_size
    slit_width = slit_width.to(u.mm).value if type(slit_width)==u.Quantity else slit_width
    slit = OpticalGroup(scene=scene, group=group, origin=origin, rotation=rotation, label=label)
    topMirror = FlatMirror(aperture=full_size, group=slit, origin=[0,0], rotation=0, label=1)
    botMirror = FlatMirror(aperture=full_size, group=slit, origin=[0,0], rotation=0, label=2)
    topMirror.extent = [0.5*slit_width, 0.5*full_size]
    botMirror.extent = [-0.5*full_size, -0.5*slit_width]
    return slit


class GeneralMirror(Mirror):
    def __init__(self,
                 R:float|u.Quantity,
                 b:float,
                 aperture:float|u.Quantity,
                 scene:Scene|None=None,
                 group=None,
                 origin:np.ndarray|list[float]=[0,0], 
                 rotation:float|u.Quantity=0*u.deg,
                 label:str = "General mirror"):
        self.R : float= R.to(u.mm).value if type(R)==u.Quantity else R
        self.aperture : float = aperture.to(u.mm).value if type(aperture)==u.Quantity else aperture
        super().__init__(equations=(lambda t: t, lambda t: (t**2/R)/(1+np.sqrt(1-(1+b)*(t/R)**2))), 
                         extent=[-0.5*self.aperture,0.5*self.aperture], 
                         scene=scene, group=group, origin=origin, rotation=rotation, label=label)

def ParabolicMirror(focal:float|u.Quantity,
                    aperture:float|u.Quantity,
                    scene:Scene|None=None,
                    group=None,
                    origin:np.ndarray|list[float]=[0,0], 
                    rotation:float|u.Quantity=0*u.deg,
                    label:str = "Parabolic mirror"):
        return GeneralMirror(R=2*focal, b=-1, aperture=aperture, scene=scene, group=group, origin=origin, rotation=rotation, label=label)

def ParabolicMirrorHole(focal:float|u.Quantity,
                        aperture:float|u.Quantity,
                        hole:float|u.Quantity,
                        scene:Scene|None=None,
                        group=None,
                        origin:np.ndarray|list[float]=[0,0], 
                        rotation:float|u.Quantity=0*u.deg,
                        label:str = "Parabolic mirror"):
    hole = hole.to(u.mm).value if type(hole)==u.Quantity else hole
    mirror = OpticalGroup(scene=scene, group=group, origin=origin, rotation=rotation, label=label)
    topMirror = ParabolicMirror(focal=focal, aperture=aperture, group=mirror, origin=[0,0], rotation=0, label=1)
    botMirror = ParabolicMirror(focal=focal, aperture=aperture, group=mirror, origin=[0,0], rotation=0, label=2)
    topMirror.extent = [0.5*hole, 0.5*aperture]
    botMirror.extent = [-0.5*aperture, -0.5*hole]
    return mirror

def HyperbolicMirror(radius:float|u.Quantity,
                     aperture:float|u.Quantity,
                     b:float=-2,
                     scene:Scene|None=None,
                     group=None,
                     origin:np.ndarray|list[float]=[0,0], 
                     rotation:float|u.Quantity=0*u.deg,
                     label:str = "Hyperbolic mirror"):
        return GeneralMirror(R=radius, b=b, aperture=aperture, scene=scene, group=group, origin=origin, rotation=rotation, label=label)

def HyperbolicMirrorHole(radius:float|u.Quantity,
                         aperture:float|u.Quantity,
                         hole:float|u.Quantity,
                         b:float=-2, 
                         scene:Scene|None=None,
                         group=None,
                         origin:np.ndarray|list[float]=[0,0], 
                         rotation:float|u.Quantity=0*u.deg,
                         label:str = "Hyperbolic mirror"):
    hole = hole.to(u.mm).value if type(hole)==u.Quantity else hole
    mirror = OpticalGroup(scene=scene, group=group, origin=origin, rotation=rotation, label=label)
    topMirror = HyperbolicMirror(radius=radius, aperture=aperture, b=b, group=mirror, origin=[0,0], rotation=0, label=1)
    botMirror = HyperbolicMirror(radius=radius, aperture=aperture, b=b, group=mirror, origin=[0,0], rotation=0, label=2)
    topMirror.extent = [0.5*hole, 0.5*aperture]
    botMirror.extent = [-0.5*aperture, -0.5*hole]
    return mirror

def SphericalMirror(radius:float|u.Quantity,
                    aperture:float|u.Quantity,
                    scene:Scene|None=None,
                    group=None,
                    origin:np.ndarray|list[float]=[0,0], 
                    rotation:float|u.Quantity=0*u.deg,
                    label:str = "Spherical mirror"):
        return GeneralMirror(R=radius, b=0, aperture=aperture, scene=scene, group=group, origin=origin, rotation=rotation, label=label)
        
def SphericalMirrorHole(radius:float|u.Quantity,
                        aperture:float|u.Quantity,
                        hole:float|u.Quantity, 
                        scene:Scene|None=None,
                        group=None,
                        origin:np.ndarray|list[float]=[0,0], 
                        rotation:float|u.Quantity=0*u.deg,
                        label:str = "Hyperbolic mirror"):
    hole = hole.to(u.mm).value if type(hole)==u.Quantity else hole
    mirror = OpticalGroup(scene=scene, group=group, origin=origin, rotation=rotation, label=label)
    topMirror = SphericalMirror(radius=radius, aperture=aperture, group=mirror, origin=[0,0], rotation=0, label=1)
    botMirror = SphericalMirror(radius=radius, aperture=aperture, group=mirror, origin=[0,0], rotation=0, label=2)
    topMirror.extent = [0.5*hole, 0.5*aperture]
    botMirror.extent = [-0.5*aperture, -0.5*hole]
    return mirror

### OPTIC.LENS ###

class Refraction(Optic):
    def __init__(self,
                 equations:list, 
                 extent:list[float],
                 n_in=1.0, n_out=1.0,
                 scene:Scene|None=None,
                 group=None,
                 origin:np.ndarray|list[float]=[0,0], 
                 rotation:float|u.Quantity=0*u.deg,
                 label:str|None = "Refraction"):
        self.n_in = n_in
        self.n_out = n_out
        super().__init__(equations, extent, scene, group, origin, rotation, label)

    def interaction(self, angle:u.Quantity, ray:Ray, t_opt:float) -> u.Quantity:
        # Dispersive lens
        if type(self.n_in)==float:
            opt_n_in = self.n_in
        else:
            opt_n_in = self.n_in(ray.wavelength)
        if type(self.n_out)==float:
            opt_n_out = self.n_out
        else:
            opt_n_out = self.n_out(ray.wavelength)
        # Entry or exit
        if ray.n_in==opt_n_in:
            n1, n2 = opt_n_in, opt_n_out
            ray.n_in = opt_n_out
        elif ray.n_in==opt_n_out:
            n1, n2 = opt_n_out, opt_n_in
            ray.n_in = opt_n_in
        else:
            print(f"WARNING : Ray ({ray.label}) entering a non-planned refraction interface ({self.label}) !")
            n1, n2 = opt_n_in, opt_n_out
        # Total internal reflection
        if abs((n1/n2)*np.sin(angle).value) > 1:
            ray.n_in = n1
            return -angle
        # Snell's law
        return (np.pi+np.arcsin((n1/n2)*np.sin(angle).value))*u.rad

class FlatRefraction(Refraction):
    def __init__(self,
                 aperture:float|u.Quantity,
                 n_in=1.0, n_out=1.0,
                 scene:Scene|None=None,
                 group=None,
                 origin:np.ndarray|list[float]=[0,0], 
                 rotation:float|u.Quantity=0*u.deg,
                 label:str|None = "Flat refraction"):
        self.aperture : float = aperture.to(u.mm).value if type(aperture)==u.Quantity else aperture
        super().__init__(equations=(lambda t: t, lambda t: 0*t), 
                         extent=[-0.5*self.aperture,0.5*self.aperture], 
                         n_in=n_in, n_out=n_out, 
                         scene=scene, group=group, origin=origin, rotation=rotation, label=label)

class CircularRefraction(Refraction):
    def __init__(self,
                 radius:float|u.Quantity,
                 aperture:float|u.Quantity,
                 n_in=1.0, n_out=1.0,
                 scene:Scene|None=None,
                 group=None,
                 origin:np.ndarray|list[float]=[0,0], 
                 rotation:float|u.Quantity=0*u.deg,
                 label:str|None = "Circular refraction"):
        self.radius : float= radius.to(u.mm).value if type(radius)==u.Quantity else radius
        self.aperture : float = aperture.to(u.mm).value if type(aperture)==u.Quantity else aperture
        super().__init__(equations=(lambda t: self.radius*np.sin(t), lambda t: -self.radius*(np.cos(np.arcsin(0.5*self.aperture/self.radius))-np.cos(t))), 
                         extent=[-np.arcsin(0.5*self.aperture/self.radius), np.arcsin(0.5*self.aperture/self.radius)], 
                         n_in=n_in, n_out=n_out, 
                         scene=scene, group=group, origin=origin, rotation=rotation, label=label)
        
def Lens(radius1:float|u.Quantity,
         radius2:float|u.Quantity,
         aperture:float|u.Quantity,
         n,
         mid_width:float|u.Quantity=0,
         scene:Scene|None=None,
         group=None,
         origin:np.ndarray|list[float]=[0,0], 
         rotation:float|u.Quantity=0*u.deg,
         label:str|None = "Lens"):
    radius1 : float= radius1.to(u.mm).value if type(radius1)==u.Quantity else radius1
    radius2 : float= radius2.to(u.mm).value if type(radius2)==u.Quantity else radius2
    aperture : float = aperture.to(u.mm).value if type(aperture)==u.Quantity else aperture
    mid_width : float= mid_width.to(u.mm).value if type(mid_width)==u.Quantity else mid_width
    lens = OpticalGroup(scene=scene, group=group, origin=origin, rotation=rotation, label=label)
    if radius1==np.inf:
        front = FlatRefraction(aperture=aperture, n_in=1.0, n_out=n, group=lens, origin=[0,0.5*mid_width], rotation=0, label=f"{label} front")
    else:
        front = CircularRefraction(radius=radius1, aperture=aperture, n_in=1.0, n_out=n, group=lens, origin=[0,0.5*mid_width], rotation=0, label=f"{label} front")
    if radius2==np.inf:
        back = FlatRefraction(aperture=aperture, n_in=1.0, n_out=n, group=lens, origin=[0,-0.5*mid_width], rotation=180, label=f"{label} back")
    else:
        back = CircularRefraction(radius=radius2, aperture=aperture, n_in=1.0, n_out=n, group=lens, origin=[0,-0.5*mid_width], rotation=180, label=f"{label} back")
    if mid_width!=0:
        top    = FlatRefraction(aperture=mid_width, n_in=1.0, n_out=n, group=lens, origin=[0.5*aperture,0], rotation=90, label=f"{label} top")
        bottom = FlatRefraction(aperture=mid_width, n_in=1.0, n_out=n, group=lens, origin=[-0.5*aperture,0], rotation=90, label=f"{label} bottom")
    return lens

def ThinSymmetricalLens(focal:float|u.Quantity,
                        aperture:float|u.Quantity,
                        n,
                        width:float=0,
                        f_wavelength:float|u.Quantity=520,
                        scene:Scene|None=None,
                        group=None,
                        origin:np.ndarray|list[float]=[0,0], 
                        rotation:float|u.Quantity=0*u.deg,
                        label:str = "Lens"):
    focal : float= focal.to(u.mm).value if type(focal)==u.Quantity else focal
    aperture : float = aperture.to(u.mm).value if type(aperture)==u.Quantity else aperture
    f_wavelength : u.Quantity = f_wavelength if type(f_wavelength)==u.Quantity else f_wavelength*u.nm
    n_f : float = n if type(n)==float else n(f_wavelength)
    radius = 2*focal*(n_f-1)
    d = abs(radius*(1-np.cos(np.arcsin(0.5*aperture/radius))))
    mid_width = max(0,width-2*d)
    if radius<0:
        mid_width = max(2*d+width,2*1.5*d)
    return Lens(radius1=radius, radius2=radius, aperture=aperture, n=n, mid_width=mid_width, scene=scene, group=group, origin=origin, rotation=rotation, label=label)

def ThinBackFlatLens(focal:float|u.Quantity,
                    aperture:float|u.Quantity,
                    n,
                    f_wavelength:float|u.Quantity=520,
                    scene:Scene|None=None,
                    group=None,
                    origin:np.ndarray|list[float]=[0,0], 
                    rotation:float|u.Quantity=0*u.deg,
                    label:str = "Lens"):
    focal : float= focal.to(u.mm).value if type(focal)==u.Quantity else focal
    aperture : float = aperture.to(u.mm).value if type(aperture)==u.Quantity else aperture
    f_wavelength : u.Quantity = f_wavelength if type(f_wavelength)==u.Quantity else f_wavelength*u.nm
    n_f : float = n if type(n)==float else n(f_wavelength)
    radius = focal*(n_f-1)
    return Lens(radius1=radius, radius2=np.inf, aperture=aperture, n=n, mid_width=0, scene=scene, group=group, origin=origin, rotation=rotation, label=label)

def ThinFrontFlatLens(focal:float|u.Quantity,
                      aperture:float|u.Quantity,
                      n,
                      f_wavelength:float|u.Quantity=520,
                      scene:Scene|None=None,
                      group=None,
                      origin:np.ndarray|list[float]=[0,0], 
                      rotation:float|u.Quantity=0*u.deg,
                      label:str = "Lens"):
    focal : float= focal.to(u.mm).value if type(focal)==u.Quantity else focal
    aperture : float = aperture.to(u.mm).value if type(aperture)==u.Quantity else aperture
    f_wavelength : u.Quantity = f_wavelength if type(f_wavelength)==u.Quantity else f_wavelength*u.nm
    n_f : float = n if type(n)==float else n(f_wavelength)
    radius = focal*(n_f-1)
    return Lens(radius1=np.inf, radius2=radius, aperture=aperture, n=n, mid_width=0, scene=scene, group=group, origin=origin, rotation=rotation, label=label)

def Prism(size:float|u.Quantity,
          n,
          angles:list[float|u.Quantity]=[60*u.deg,60*u.deg],
          scene:Scene|None=None,
          group=None,
          origin:np.ndarray|list[float]=[0,0], 
          rotation:float|u.Quantity=0*u.deg,
          label:str = "Prism"):
    size : float = size.to(u.mm).value if type(size)==u.Quantity else size
    for i,angle in enumerate(angles):
        angles[i] = angle if type(angle)==u.Quantity else angle*u.deg
    prism = OpticalGroup(scene=scene, group=group, origin=origin, rotation=rotation, label=label)
    sin_ratio = size/np.sin(180*u.deg-angles[0]-angles[1]).value
    size_l = sin_ratio*np.sin(angles[1]).value
    size_r = sin_ratio*np.sin(angles[0]).value
    A = np.array([0,0])
    B = np.array([size,0])
    C = A + np.array([size_l*np.cos(angles[0]), size_l*np.sin(angles[0])])
    center = (A+B+C)/3
    center_b = (A+B)/2
    center_l = (A+C)/2
    center_r = (B+C)/2
    base  = FlatRefraction(aperture=size,   n_in=1.0, n_out=n, group=prism, origin=center_b-center, rotation=0, label=f"Base")
    left  = FlatRefraction(aperture=size_l, n_in=1.0, n_out=n, group=prism, origin=center_l-center, rotation=angles[0], label=f"Left")
    right = FlatRefraction(aperture=size_r, n_in=1.0, n_out=n, group=prism, origin=center_r-center,  rotation=180*u.deg-angles[1], label=f"Right")
    return prism

class SchmidtRefraction(Refraction):
    def __init__(self,
                 aperture:float|u.Quantity,
                 radius:float|u.Quantity,
                 n,
                 k:float=1.5,
                 opt_wavelength:float|u.Quantity=520*u.nm,
                 scene:Scene|None=None,
                 group=None,
                 origin:np.ndarray|list[float]=[0,0], 
                 rotation:float|u.Quantity=0*u.deg,
                 label:str|None = "Schmidt refraction"):
        self.aperture : float = aperture.to(u.mm).value if type(aperture)==u.Quantity else aperture
        self.radius : float = radius.to(u.mm).value if type(radius)==u.Quantity else radius
        self.n = n
        self.k = k
        if type(self.n)==float:
            opt_n = self.n
        else:
            opt_n = self.n(opt_wavelength)
        super().__init__(equations=(lambda t: t, lambda t: (0.25*k*(self.aperture*t)**2 - t**4)/(4*(opt_n-1)*self.radius**3)), 
                         extent=[-0.5*self.aperture,0.5*self.aperture], 
                         n_in=1.0, n_out=self.n, 
                         scene=scene, group=group, origin=origin, rotation=rotation, label=label)
        
def SchmidtCorrector(aperture:float|u.Quantity,
                     radius:float|u.Quantity,
                     n:float,
                     k:float=1.5,
                     width:float|u.Quantity=1,
                     opt_wavelength:float|u.Quantity=520*u.nm,
                     scene:Scene|None=None,
                     group=None,
                     origin:np.ndarray|list[float]=[0,0], 
                     rotation:float|u.Quantity=0*u.deg,
                     label:str|None = "Schmidt corrector"):
    aperture : float = aperture.to(u.mm).value if type(aperture)==u.Quantity else aperture
    radius : float = radius.to(u.mm).value if type(radius)==u.Quantity else radius
    width : float = width.to(u.mm).value if type(width)==u.Quantity else width
    corrector = OpticalGroup(scene=scene, group=group, origin=origin, rotation=rotation, label=label)
    schmidt = SchmidtRefraction(aperture=aperture, radius=radius, n=n, k=k, opt_wavelength=opt_wavelength, group=corrector, origin=[0,-width], rotation=0, label="Schmidt")
    flat = FlatRefraction(aperture=aperture, n_in=1.0, n_out=n, group=corrector, origin=[0,0], rotation=0, label="Flat")
    real_width = width-schmidt.raw_equation_y(0.5*aperture)
    left = FlatRefraction(aperture=real_width, n_in=1.0, n_out=n, group=corrector, origin=[-0.5*aperture, -0.5*real_width], rotation=90, label="Left")
    right = FlatRefraction(aperture=real_width, n_in=1.0, n_out=n, group=corrector, origin=[0.5*aperture, -0.5*real_width], rotation=90, label="Right")
    return corrector

### OPTIC.GRATING ###

class Grating(FlatMirror):
    def __init__(self,
                 aperture:u.Quantity|float,
                 period:u.Quantity|float,
                 order:int=1, 
                 scene:Scene|None=None,
                 group=None, 
                 origin:np.ndarray|list[float]=[0,0], 
                 rotation:float|u.Quantity=0*u.deg,
                 label:str = "Grating"):
        self.period : u.Quantity = period if type(period)==u.Quantity else period/u.mm
        self.order : int = order
        self.aperture : float = aperture.to(u.mm).value if type(aperture)==u.Quantity else aperture
        super().__init__(aperture, scene, group, origin, rotation, label)

    def interaction(self, angle:u.Quantity, ray:Ray, t_opt:float) -> u.Quantity:
        return np.arcsin((self.order*ray.wavelength*self.period).to(u.dimensionless_unscaled).value - np.sin(angle).value)*u.rad

class TransmissionGrating(FlatMirror):
    def __init__(self,
                 aperture:u.Quantity|float,
                 period:u.Quantity|float,
                 order:int=1, 
                 scene:Scene|None=None,
                 group=None, 
                 origin:np.ndarray|list[float]=[0,0], 
                 rotation:float|u.Quantity=0*u.deg,
                 label:str = "Transmission grating"):
        self.period : u.Quantity = period if type(period)==u.Quantity else period/u.mm
        self.order : int = order
        self.aperture : float = aperture.to(u.mm).value if type(aperture)==u.Quantity else aperture
        super().__init__(aperture, scene, group, origin, rotation, label)

    def interaction(self, angle:u.Quantity, ray:Ray, t_opt:float) -> u.Quantity:
        return (np.pi+np.arcsin((self.order*ray.wavelength*self.period).to(u.dimensionless_unscaled).value - np.sin(angle).value))*u.rad


### VISUALIZATION ###

class Camera(FlatMirror):
    def __init__(self,
                 aperture:u.Quantity|float,
                 scene:Scene|None=None,
                 group=None, 
                 origin:np.ndarray|list[float]=[0,0], 
                 rotation:float|u.Quantity=0*u.deg,
                 label:str = "Camera"):
        self.hits : list[float] = []
        self.colors : list[list[float]] = []
        self.aperture : float = aperture.to(u.mm).value if type(aperture)==u.Quantity else aperture
        super().__init__(aperture, scene, group, origin, rotation, label)

    def interaction(self, angle:u.Quantity, ray:Ray, t_opt:float) -> u.Quantity:
        self.hits.append(t_opt)
        self.colors.append(ray.color)
        return None
    
    def plot(self, ax=None):
        ax = plt.subplots(figsize=(8,2))[1] if ax is None else ax
        ax.scatter(self.hits, [0]*len(self.hits), c=self.colors, s=1)
        ax.set_xlim(self.extent[0], self.extent[1])
        plt.show()
    

### GROUPS ###

class OpticalGroup():
    def __init__(self,
                 elements:list=[],
                 scene:Scene|None=None,
                 group=None,
                 origin:np.ndarray|list[float]=[0,0],
                 rotation:float|u.Quantity=0*u.deg,
                 label:str|None=None):
        self.scene : Scene = scene
        self.group : OpticalGroup = group
        self.origin : np.ndarray = origin if type(origin)==np.ndarray else np.array(origin)
        self.rotation = rotation if type(rotation)==u.Quantity else rotation*u.deg
        self.label = label
        self.elements : list = []
        for elem in elements:
            self.append(elem)

    def append(self, elem:Optic) -> None:
        self.elements.append(elem)
        new_origin = [self.origin[0] + elem.origin[0]*np.cos(self.rotation).value - elem.origin[1]*np.sin(self.rotation).value,
                      self.origin[1] + elem.origin[0]*np.sin(self.rotation).value + elem.origin[1]*np.cos(self.rotation).value]
        new_rotation = elem.rotation + self.rotation
        elem.update_pos(new_origin, new_rotation)
        elem.label = f"{self.label}/{elem.label}"
        if (self.scene is not None) & (self.group is None): self.scene.append(elem)
        if self.group is not None: self.group.append(elem)

    def update_pos(self, new_pos:list[float]=None, new_rot:u.Quantity|float=None):
        new_pos = self.origin if new_pos is None else new_pos
        new_rot = self.rotation if new_rot is None else new_rot
        new_pos = new_pos if type(new_pos)==np.ndarray else np.array(new_pos)
        new_rot = new_rot if type(new_rot)==u.Quantity else new_rot*u.deg
        self.origin = new_pos
        self.rotation = new_rot
        for elem in self.elements:
            new_origin = [self.origin[0] + elem.origin[0]*np.cos(self.rotation).value - elem.origin[1]*np.sin(self.rotation).value,
                      self.origin[1] + elem.origin[0]*np.sin(self.rotation).value + elem.origin[1]*np.cos(self.rotation).value]
            new_rotation = elem.rotation + self.rotation
            elem.update_pos(new_origin, new_rotation)

    def merge(self, other):
        scene = None if self.scene!=other.scene else self.scene
        origin = self.origin
        rotation = self.rotation
        label = f"{self.label}+{other.label}"
        elements = self.elements + other.elements
        if scene is not None:
            for elem in elements:
                if elem in scene.rays: scene.rays.remove(elem)
                if elem in scene.optics: scene.optics.remove(elem)
        group = OpticalGroup(elements, scene, origin, rotation, label)
        return group
    
    def __add__(self, other):
        return self.merge(other)
        
def CassegrainTelescope(focal:float,
                        aperture:float,
                        backfocus:float,
                        length:float,
                        scene:Scene|None=None,
                        group=None, 
                        origin:np.ndarray|list[float]=[0,0], 
                        rotation:float|u.Quantity=0*u.deg,
                        label:str = "Cassegrain",
                        return_geometry:bool=False):
    q = length+backfocus
    M = (focal-q)/(q-backfocus)
    f1 = focal/M
    p = (f1+backfocus)/(M+1)
    aperture2 = aperture*p/f1
    r2 = 2*q/(M-1)
    b = -(4*M)/(M-1)**2 -1
    telescope = OpticalGroup(scene=scene, origin=origin, rotation=rotation, label=label)
    primary = ParabolicMirrorHole(focal=f1, aperture=aperture, hole=aperture2, group=telescope, origin=[backfocus,0], rotation=-90)
    secondary = HyperbolicMirror(radius=r2, aperture=aperture2, b=b, group=telescope, origin=[q,0], rotation=-90)
    if return_geometry:
        geometry = {'Primary':{'Diameter':aperture,'Focal':f1}, 'Secondary':{'Diameter':aperture2,'Curvature':r2,'Conic':b}, 'Backfocus':backfocus}
        return telescope, geometry
    return telescope

def RitcheyChretienTelescope(focal:float,
                             aperture:float,
                             backfocus:float,
                             length:float,
                             scene:Scene|None=None,
                             group=None, 
                             origin:np.ndarray|list[float]=[0,0], 
                             rotation:float|u.Quantity=0*u.deg,
                             label:str = "Ritech-Chrétien",
                             return_geometry:bool=False):
    q = length+backfocus
    d = q-backfocus
    s = q/d
    M = (focal-q)/(q-backfocus)
    r1 = 2*focal/M
    r2 = 2*q/(M-1)
    p = (r1+backfocus)/(M+1)
    aperture2 = aperture*p/r1
    b1 = -2*s/M**3 -1
    b2 = (-4*M*(M-1) - 2*(M+s))/(M-1)**3 - 1
    telescope = OpticalGroup(scene=scene, origin=origin, rotation=rotation, label=label)
    primary = HyperbolicMirrorHole(radius=r1, aperture=aperture, b=b1, hole=aperture2, group=telescope, origin=[backfocus,0], rotation=-90)
    secondary = HyperbolicMirror(radius=r2, aperture=aperture2, b=b2, group=telescope, origin=[q,0], rotation=-90)
    if return_geometry:
        geometry = {'Primary':{'Diameter':aperture,'Curvature':r1,'Conic':b1}, 'Secondary':{'Diameter':aperture2,'Curvature':r2,'Conic':b2}, 'Backfocus':backfocus}
        return telescope, geometry
    return telescope