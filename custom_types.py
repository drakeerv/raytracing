import typing
import numpy
import util
import math

class Material:
    def __init__(self, color: numpy.ndarray, emission_color: numpy.ndarray, specular_color: numpy.ndarray, smoothness: float, specular: float, emission_strength: float) -> None:
        self.color = color
        self.emission_color = emission_color
        self.specular_color = specular_color
        self.smoothness = smoothness
        self.specular = specular
        self.emission_strength = emission_strength

class GameObject:
    def __init__(self, position: numpy.ndarray, rotation: numpy.ndarray) -> None:
        self.position = position
        self.rotation = rotation

class ObjectsHandler:
    def __init__(self) -> None:
        self.objects = []

    def add(self, obj: GameObject) -> None:
        self.objects.append(obj)

    def get(self, index: int) -> GameObject:
        return self.objects[index]

    def remove(self, index: int) -> None:
        self.objects.pop(index)

    def count(self) -> int:
        return len(self.objects)
    
    def clear(self) -> None:
        self.objects.clear()

    def __iter__(self):
        return iter(self.objects)

class PhysicalObject(GameObject):
    def __init__(self, position: numpy.ndarray, rotation: numpy.ndarray, material: Material) -> None:
        super().__init__(position, rotation)
        self.material = material

    def intersect(self, ray) -> typing.Tuple[float, GameObject]:
        return None, None

    def get_normal(self) -> numpy.ndarray:
        return None
    
class Ray:
    def __init__(self, origin: numpy.ndarray, direction: numpy.ndarray, max_distance: float, objects_handler: ObjectsHandler) -> None:
        self.origin = origin
        self.direction = direction
        self.max_distance = max_distance
        self.objects_handler = objects_handler
    
    def get_intersection(self) -> tuple[float, PhysicalObject | None]:
        min_t = self.max_distance
        min_obj = None

        for obj in self.objects_handler:
            if not isinstance(obj, PhysicalObject) or util.get_distance(self.origin, obj.position) > self.max_distance:
                continue

            t, obj = obj.intersect(self)

            if t is not None and t < min_t:
                min_t = t
                min_obj = obj

        return min_t, min_obj

    def get_point(self, distance: float) -> numpy.ndarray:
        return self.origin + distance * self.direction

class Sphere(PhysicalObject):
    def __init__(self, position: numpy.ndarray, rotation: numpy.ndarray, material: Material, radius: float) -> None:
        super().__init__(position, rotation, material)
        self.radius = radius

    def intersect(self, ray) -> typing.Tuple[float, typing.Self]:
        a = numpy.dot(ray.direction, ray.direction)
        b = 2 * numpy.dot(ray.direction, ray.origin - self.position)
        c = numpy.dot(ray.origin - self.position, ray.origin - self.position) - self.radius * self.radius

        discriminant = b * b - 4 * a * c

        if discriminant < 0:
            return None, None

        t1 = (-b + math.sqrt(discriminant)) / (2 * a)
        t2 = (-b - math.sqrt(discriminant)) / (2 * a)

        if t1 < 0 and t2 < 0:
            return None, None

        if t1 < 0:
            return t2, self

        if t2 < 0:
            return t1, self

        return min(t1, t2), self
    
    def get_normal(self, point: numpy.ndarray) -> numpy.ndarray:
        return util.normalize(point - self.position)
    
class Camera(GameObject):
    def __init__(self, position: numpy.ndarray, rotation: numpy.ndarray, objects_handler: ObjectsHandler, width: int, height: int, fov: float, max_distance: float) -> None:
        super().__init__(position, rotation)
        self.objects_handler = objects_handler
        self.width = width
        self.height = height
        self.fov = fov
        self.max_distance = max_distance

    def get_ray_direction(self, x: float, y: float) -> numpy.ndarray:
        return util.rotate(numpy.array([
            (2 * (x + 0.5) / self.width - 1) * math.tan(self.fov / 2) * self.width / self.height,
            (1 - 2 * (y + 0.5) / self.height) * math.tan(self.fov / 2),
            -1,
        ]), self.rotation)
    
    def get_ray(self, x: float, y: float,) -> Ray:
        return Ray(self.position, self.get_ray_direction(x, y), self.max_distance, self.objects_handler)