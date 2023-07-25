import json
from math import degrees, radians
from pathlib import Path
from cambrian.utils.utils import NumpyEncoder
import numpy as np
from typing import List, Tuple
from prodict import Prodict
import yaml


from cambrian.evolution_envs.eye import SinglePixel
from cambrian.utils.renderer_utils import get_sensor_plane_angles, name_to_fdir, points_on_circumference

class OculozoicAnimal:
    def __init__(self, config):
        self.config = config
        self.max_photoreceptors = config.max_photoreceptors
        self.mutation_count = 0 
        self.num_photoreceptors = self.config.init_photoreceptors
        self.pixels = []

        # constants 
        self.mutation_types = ['add_photoreceptor', 'simple_to_lens', 'simple_to_lens', 'update_pixel']
        self.radius = self.config.radius
        self.max_num_eyes_per_side = self.config.max_num_eyes_per_side
        self.visual_acuity_sigma = self.config.visual_acuity_sigma

    def init_animal(self, mx, my):
        self.reset_position(mx, my)
        self.num_pixels = 1
        # Initialize positions on the eye
        self.right_eye_pixels = points_on_circumference(center = self.position, r= self.radius, n = self.max_num_eyes_per_side, direction='right')
        self.right_eye_pixels_occupancy = np.zeros(len(self.right_eye_pixels))
        self.right_angles = get_sensor_plane_angles(self.position, self.left_eye_pixels)
        self.left_eye_pixels = points_on_circumference(center = self.position, r= self.radius, n = self.max_num_eyes_per_side, direction='left')
        self.left_eye_pixels_occupancy = np.zeros(len(self.left_eye_pixels))
        self.left_angles = get_sensor_plane_angles(self.position, self.left_eye_pixels)

        imaging_model = 'simple'
        fov = 120
        _default_idx = -1 
        pixel_idx = len(self.left_eye_pixels)
        angle = 95. #self.left_angles[_default_idx] #0. #60
        self.sensor_size = self.config.init_sensor_size # large sensor size 
        self.left_eye_pixels_occupancy[_default_idx] = 1
        pixel_pos = self.left_eye_pixels[_default_idx]
        pixel_config = self.generate_pixel_config(imaging_model, fov, angle, direction='left', pixel_pos=pixel_pos, pixel_idx = pixel_idx) # should be looking left down

        # self.right_eye_pixels_occupancy[_default_idx] = 1
        # pixel_pos = self.right_eye_pixels[_default_idx]
        # pixel_config = self.generate_pixel_config(imaging_model, fov, angle, direction='right', 
        #                                           pixel_pos=pixel_pos, pixel_idx = pixel_idx) # should be looking left down
        self.add_pixel(pixel_config)
        # self.mutation_count += 1
        self.mutation_chain = []

    def observe_scene(self, dx, dy, geometry):
        """
        processed_eye_intensity: List of Intensity outputs (per eye)
            [21, 39, 28, 9, 1...]
        eye_out: List of photoreceptors outputs per eye
            [raw_photoreceptor_1, raw_photoreceptor_2, ..., raw_photoreceptor_n]
        """
        self._update_position(dx, dy)
        eye_out = []
        processed_eye_intensity = []
        num_photoreceptors_per_pixel = np.maximum(0, int(self.num_photoreceptors/self.num_pixels))
        for i in range(self.num_pixels):
            final_intensity, raw_photoreceptor_output = self.pixels[i].render_pixel(dx, dy, 
                                                                   geometry,
                                                                    num_photoreceptors_per_pixel)
            if False:
            # if self.config.total_intensity_output_only:
                intensity, _ = np.mean(raw_photoreceptor_output)
                eye_out.append(intensity)
            else:
                eye_out.append(raw_photoreceptor_output)
                processed_eye_intensity.append(final_intensity)

        return processed_eye_intensity, eye_out
    
    def reset_position(self, mx, my):
        self.x = mx
        self.y = my
        self.position = np.array([self.x, self.y])
        self.right_eye_pixels = points_on_circumference(center = self.position, r= self.radius, n = self.max_num_eyes_per_side, direction='right')
        self.left_eye_pixels = points_on_circumference(center = self.position, r= self.radius, n = self.max_num_eyes_per_side, direction='left')

    def mutate(self, mutation_type, mut_args=Prodict):
        self.mutation_count += 1
        _mut = Prodict()
        _mut.type = mutation_type

        if mutation_type == 'add_photoreceptor':
            self.num_photoreceptors += self.config.increment_photoreceptor
            self.num_photoreceptors = np.clip(self.num_photoreceptors, 
                                              self.config.init_photoreceptors, self.max_photoreceptors)
            _mut.args = {'num_photoreceptors': self.num_photoreceptors}

        elif mutation_type == 'simple_to_lens':
            if mut_args.pixel_idx == None: 
                # start from the first compound eye and incrementally add a lens, break after affing 
                for i in range(self.num_pixels):
                    if self.pixels[i].imaging_model == 'simple':
                        # the default imaging model will be used (at init)
                        self.pixels[i].imaging_model = 'lens'
                        _mut.args = {'cam_idx': i}
                        break
            else:
                self.pixels[mut_args.pixel_idx].imaging_model = 'lens'
                
            _mut.args = {'cam_idx': mut_args.pixel_idx}

        elif mutation_type == 'add_pixel':
            _dir = np.random.choice('left', 'right')
            pixel_config = self.generate_pixel_config(mut_args.imaging_model, mut_args.fov, mut_args.angle, pixel_pos=None, direction=_dir)
            self.add_pixel(pixel_config)
            _mut.args = {'imaging_model': mut_args.imaging_model, 'fov': mut_args.fov, 'angle': mut_args.angle, 
                         'pixel_pos': None, 'direction': _dir}

        elif mutation_type == 'update_pixel':
            if mut_args.pixel_idx == None: 
                # mut_args.pixel_idx = np.random.choice(np.where(self.left_eye_pixels_occupancy.ravel()==1)[0])
                # mut_args.pixel_idx = np.random.choice(np.where(self.right_eye_pixels_occupancy.ravel()==1)[0])
                mut_args.pixel_idx = np.random.choice(np.arange(0, len(self.pixels)))

            self.pixels[mut_args.pixel_idx].update_pixel_config(mut_args.fov_r_update, mut_args.angel_r_update, mut_args.sensor_update)
            _mut.args = {'fov_update':mut_args.fov_r_update, 
                         'angel_r_update': mut_args.angel_r_update, 
                         'sensor_update': mut_args.sensor_update
                         }
        else:
            raise ValueError("{} not found".format(mutation_type))
        
        self.mutation_chain.append(_mut)

    def add_pixel(self, pixel_config):
        if self.num_photoreceptors < self.num_pixels: 
            # atleast one photoreceptor per pixel
            # print("atleast one photoreceptor per pixel")
            return 
        pixel = SinglePixel(pixel_config)
        self.pixels.append(pixel)
        self.num_pixels = len(self.pixels)

    def remove_pixel(self, ):
        self.num_pixels -= 1
        self.pixels.pop()

    def generate_pixel_config(self, imaging_model, fov, angle, direction, pixel_pos=None, pixel_idx=None):
        """
        Generate a pixel configuration. The pixel config takes in eye properties 
        and a pixel position and index on the animal. If none, it will sample the closest 
        idx available from the ones that are occupied. 
        """
        config = Prodict()
        if pixel_pos is None: 
            pixel_pos, pixel_idx = self._sample_new_pixel_position(direction)

        config.x = pixel_pos[0]
        config.y = pixel_pos[1]
        config.sensor_size = self.sensor_size
        config.fov = fov
        config.angle = angle
        config.animal_direction = direction
        config.animal_idx = pixel_idx
        num_photoreceptprs_per_pixel = int(self.num_photoreceptors/self.num_pixels)
        config.visual_acuity = num_photoreceptprs_per_pixel/self.visual_acuity_sigma
        config.imaging_model = imaging_model
        config.diffuse_sweep_rays = self.config.diffuse_sweep_rays
        return config 

    def load_animal_from_state(self, state_config_file):

        with open(state_config_file, "r") as ymlfile:
            dict_cfg = yaml.load(ymlfile, Loader=yaml.Loader)
            _state = Prodict.from_dict(dict_cfg)

        # load animal constant
        self.radius = _state.radius
        self.max_num_eyes_per_side = _state.max_num_pixels_per_side
        self.visual_acuity_sigma = _state.visual_acuity_sigma
        _state.num_photoreceptors = self.num_photoreceptors
        self.mutation_count = _state.mutation_count
        self.mutation_types = _state.mutation_types
        self.mutation_chain = _state.mutation_chain

        # load eye location
        # self.right_eye_pixels = _state.right_eye_pixels # right eye pixel positions can be reset based on position and radius, only the occupancy matters
        # self.left_eye_pixels = _state.left_eye_pixels
        self.right_eye_pixels_occupancy = _state.right_eye_pixels_occupancy
        self.left_eye_pixels_occupancy = _state.left_eye_pixels_occupancy

        # load pixel configuration
        self.num_pixels = _state.num_pixels
        self.pixels = []
        for p_config in _state.pixel_configs: 
            pixel = SinglePixel()
            pixel.load_from_config(p_config)
            self.pixels.append(pixel.from_dict())

    def save_animal_state(self, save_dir=None):
        """
        if save_dir is None, it just returns the state
        """
        _state = {}
        _state = Prodict.from_dict(_state)
        
        # save animal constant
        _state.radius = self.radius
        _state.max_num_pixels_per_side = self.max_num_eyes_per_side
        _state.visual_acuity_sigma = self.visual_acuity_sigma
        _state.num_photoreceptors = self.num_photoreceptors
        _state.mutation_count = self.mutation_count
        _state.mutation_types = self.mutation_types
        _state.mutation_chain = self.mutation_chain

        # save eye location: right eye pixel positions can be reset based on position and radius, only the occupancy matters
        # _state.right_eye_pixels = self.right_eye_pixels
        # _state.left_eye_pixels = self.left_eye_pixels
        _state.right_eye_pixels_occupancy = self.right_eye_pixels_occupancy
        _state.left_eye_pixels_occupancy = self.left_eye_pixels_occupancy

        # save pixel configuration
        _state.num_pixels = len(self.pixels)
        _state.pixel_configs = []
        for p in self.pixels: 
            _state.pixel_configs.append(p.get_config_state().to_dict())

        if save_dir is not None: 
            target = str(Path.joinpath(save_dir,'mutation_{}.json'.format(self.mutation_count)))
            with open(target, 'w') as outfile:
                _state = _state.to_dict()
                # yaml.dump(_state, outfile, default_flow_style=False)
                json.dump(_state, outfile, indent = 6, cls=NumpyEncoder)

        return _state

    def _update_position(self, dx, dy):
        self.x += dx
        self.y += dy
        self.position = np.array([[self.x], [self.y]])
        # everything else will shift equally as well. 
        self.right_eye_pixels += np.array([dx, dy])
        self.left_eye_pixels += np.array([dx, dy])

    def _sample_new_pixel_position(self, direction='left'):
        if direction == 'left':
            idx = get_idx(self.left_eye_pixels_occupancy)
            if idx is None: 
                return None 
            pixel_pos = self.left_eye_pixels[idx]
            self.left_eye_pixels_occupancy[idx] = 1
        else:
            idx = get_idx(self.right_eye_pixels_occupancy)
            if idx is None: 
                return None 
            pixel_pos = self.right_eye_pixels[idx]
            self.right_eye_pixels_occupancy[idx] = 1

        return pixel_pos, idx

    def print_state(self, ):
        print('--------------------')
        print("Animal has {} eyes with {} photoreceptors.".format(self.num_pixels, self.num_photoreceptors))
        for i, p in enumerate(self.pixels): 
            print(
                "Eye {} with type {} looking {}, fov: {}, angle: {}, sensor size: {}.".format(
                    i, p.imaging_model, p.animal_direction, degrees(p.fov_r), degrees(p.angle_r), p.sensor_size
                )
            )
        print('--------------------')

################ 
## Utils
################ 

def get_idx(arr):
    # start from the middle and iterate left and right to find the first empty index
    l = len(arr)
    _c = int(l/2)
    _dir = 1
    i = 0 
    empty = True
    while empty:
        idx = int(_c + (i * _dir))
        if idx < 0 or idx > len(arr):
            return None
        e = arr[idx]
        if e == 0:
            empty = False 
            return idx
        else:
            _dir *= -1
            idx = int(_c + (i * _dir))
            if idx < 0 or idx > len(arr)-1:
                return None
            e = arr[idx]
            if e == 0:
                empty = False 
                return idx

        i +=1 
    
    return idx