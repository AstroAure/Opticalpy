import autograd.numpy as np
import astropy.units as u

class OpticalGroup():
    def __init__(self,
                 elements:list=[],
                 scene=None,
                 group=None,
                 origin:np.ndarray|list[float]=[0,0],
                 rotation:float|u.Quantity=0*u.deg,
                 label:str|None=None):
        self.scene = scene
        self.group : OpticalGroup = group
        self.origin : np.ndarray = origin if type(origin)==np.ndarray else np.array(origin)
        self.rotation = rotation if type(rotation)==u.Quantity else rotation*u.deg
        self.label = label
        self.elements : list = []
        for elem in elements:
            self.append(elem)

    def append(self, elem) -> None:
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