import numpy

def get_distance(point1: numpy.ndarray, point2: numpy.ndarray) -> float:
    return numpy.linalg.norm(point1 - point2)

def rotate(point: numpy.ndarray, rotation: numpy.ndarray) -> numpy.ndarray:
    return numpy.array([
        point[0] * numpy.cos(rotation[1]) + point[2] * numpy.sin(rotation[1]),
        point[1] * numpy.cos(rotation[0]) + point[2] * numpy.sin(rotation[0]) * numpy.sin(rotation[1]) + point[0] * numpy.sin(rotation[0]) * numpy.cos(rotation[1]),
        point[2] * numpy.cos(rotation[0]) * numpy.cos(rotation[1]) - point[1] * numpy.sin(rotation[0]) + point[0] * numpy.cos(rotation[0]) * numpy.sin(rotation[1])
    ])

def reflect(direction: numpy.ndarray, normal: numpy.ndarray) -> numpy.ndarray:
    return direction - 2 * numpy.dot(direction, normal) * normal

def normalize(vector: numpy.ndarray) -> numpy.ndarray:
    return vector / numpy.linalg.norm(vector)

def rotation_matrix(rotation: numpy.ndarray) -> numpy.ndarray:
    return numpy.array([
        [numpy.cos(rotation[1]) * numpy.cos(rotation[2]), -numpy.cos(rotation[1]) * numpy.sin(rotation[2]), numpy.sin(rotation[1])],
        [numpy.sin(rotation[0]) * numpy.sin(rotation[1]) * numpy.cos(rotation[2]) + numpy.cos(rotation[0]) * numpy.sin(rotation[2]), -numpy.sin(rotation[0]) * numpy.sin(rotation[1]) * numpy.sin(rotation[2]) + numpy.cos(rotation[0]) * numpy.cos(rotation[2]), -numpy.sin(rotation[0]) * numpy.cos(rotation[1])],
        [-numpy.cos(rotation[0]) * numpy.sin(rotation[1]) * numpy.cos(rotation[2]) + numpy.sin(rotation[0]) * numpy.sin(rotation[2]), numpy.cos(rotation[0]) * numpy.sin(rotation[1]) * numpy.sin(rotation[2]) + numpy.sin(rotation[0]) * numpy.cos(rotation[2]), numpy.cos(rotation[0]) * numpy.cos(rotation[1])]
    ])

def inverse(rotation: numpy.ndarray) -> numpy.ndarray:
    return numpy.array([rotation[0], -rotation[1], -rotation[2]])

def generate_random_direction() -> numpy.ndarray:
    phi = numpy.random.uniform(0, 2 * numpy.pi)
    theta = numpy.random.uniform(0, numpy.pi)

    return numpy.array([
        numpy.sin(theta) * numpy.cos(phi),
        numpy.sin(theta) * numpy.sin(phi),
        numpy.cos(theta)
    ])

def lerp(arr1: numpy.ndarray, arr2: numpy.ndarray, t: float) -> numpy.ndarray:
    return arr1 + t * (arr2 - arr1)