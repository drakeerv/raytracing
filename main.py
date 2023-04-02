import cv2
import util
import numpy
import random
import custom_types

def generate_progress_bar(width: int, progress: float) -> str:
    progress = int(progress * width)
    return f"[{'=' * progress}{' ' * (width - progress)}]"

class Main:
    def __init__(self, width, height, name) -> None:
        self.width = width
        self.height = height
        self.name = name

        self.render_buffer = numpy.zeros((self.height, self.width, 3), numpy.uint8)

        cv2.namedWindow(self.name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.name, self.width, self.height)

        self.objects_handler = custom_types.ObjectsHandler()

        self.camera = custom_types.Camera(numpy.array([0, 0, 0]), numpy.array([0, 0, 0]), self.objects_handler, self.width, self.height, 90, 1000)
        self.objects_handler.add(self.camera)

        self.objects_handler.add(custom_types.Sphere(numpy.array([0, 0, -4]), numpy.array([0, 0, 0]), custom_types.Material(numpy.array([255, 255, 255]), numpy.array([255, 255, 255]), numpy.array([255, 255, 255]), 1.0, 1.0, 10.0), 1))
        self.objects_handler.add(custom_types.Sphere(numpy.array([0, 4, -5]), numpy.array([0, 0, 0]), custom_types.Material(numpy.array([0, 255, 0]), numpy.array([0, 0, 0]), numpy.array([255, 255, 255]), 0.5, 0.5, 0.0), 2))
        self.objects_handler.add(custom_types.Sphere(numpy.array([0, -4, -5]), numpy.array([0, 0, 0]), custom_types.Material(numpy.array([0, 0, 255]), numpy.array([0, 0, 0]), numpy.array([255, 255, 255]), 1.0, 1.0, 0.0), 2))
        # giant glowing sphere on the bottom
        self.objects_handler.add(custom_types.Sphere(numpy.array([0, 0, 25]), numpy.array([0, 0, 0]), custom_types.Material(numpy.array([255, 255, 255]), numpy.array([255, 255, 255]), numpy.array([255, 255, 255]), 1.0, 1.0, 1.0), 20))

        self.bounce_limit = 3
        self.render_frames = 200

    def render(self):
        self.render_buffer.fill(0)
        frames = []

        for frame in range(self.render_frames):
            current_frame = numpy.zeros((self.height, self.width, 3), numpy.uint8)
            for x in range(self.width):
                for y in range(self.height):
                    ray = self.camera.get_ray(x, y)

                    light = numpy.zeros(3)
                    color = numpy.ones(3)

                    for _ in range(self.bounce_limit):
                        distance, obj = ray.get_intersection()

                        if obj is None:
                            break

                        material = obj.material

                        hit_point = ray.get_point(distance)
                        hit_normal = obj.get_normal(hit_point)

                        diffused_dir = util.normalize(hit_normal + util.generate_random_direction())
                        specular_dir = util.reflect(-ray.direction, hit_normal)
                        is_specular = material.specular >= random.random()
                        final_dir = util.lerp(diffused_dir, specular_dir, material.smoothness * is_specular)

                        # calculate color based on material properties
                        emitted_light = material.emission_color * material.emission_strength
                        light += emitted_light * color
                        color *= util.lerp(material.color, material.specular_color, is_specular)

                        ray.origin = hit_point
                        ray.direction = final_dir

                    current_frame[y][x] = light

                cv2.imshow(self.name, current_frame)
                cv2.waitKey(1)

                print(f"Rendering frame {frame + 1}/{self.render_frames} {generate_progress_bar(50, x / self.width)}", end="\r")
            frames.append(current_frame)

        self.render_buffer = numpy.average(frames, axis=0).astype(numpy.uint8)

        while True:
            cv2.imshow(self.name, self.render_buffer)
            cv2.waitKey(1)

if __name__ == "__main__":
    main = Main(1920, 1080, "Raytracer")
    main.render()
