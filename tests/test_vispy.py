import unittest
from itertools import cycle
import numpy as np
from vispy import app, scene
from vispy.scene import visuals
from vispy.visuals.transforms import STTransform  # Scale-Translate
from vispy.color import BaseColormap, get_colormaps

from phathom.multivol import MultiVolume, get_translucent_cmap, RGBAVolume

import tifffile
import os
from skimage import img_as_float32
from skimage.exposure import rescale_intensity

working_dir = '/media/jswaney/Drive/Justin/organoid_etango/test_ventricle'
syto16_img = tifffile.imread(os.path.join(working_dir, 'syto16_clahe_16x.tif'))
sox2_img = tifffile.imread(os.path.join(working_dir, 'sox2_clahe_16x.tif'))
syto16_img = img_as_float32(rescale_intensity(syto16_img))
sox2_img = img_as_float32(rescale_intensity(sox2_img))


# create colormaps that work well for translucent and additive volume rendering
class TransFire(BaseColormap):
    glsl_map = """
    vec4 translucent_fire(float t) {
        return vec4(pow(t, 0.5), t, t*t, max(0, t*1.05 - 0.05));
    }
    """

    def __repr__(self):
        return "TransFire"


class TransGrays(BaseColormap):
    glsl_map = """
    vec4 translucent_grays(float t) {
        return vec4(t, t, t, t*0.5);
    }
    """

    def __repr__(self):
        return "TransGrays"


class TransRed(BaseColormap):
    glsl_map = """
    vec4 translucent_grays(float t) {
        return vec4(t, 0, 0, t*0.5);
    }
    """

    def __repr__(self):
        return "TransRed"


class TransBlue(BaseColormap):
    glsl_map = """
    vec4 translucent_grays(float t) {
        return vec4(0, 0, t, t*0.5);
    }
    """

    def __repr__(self):
        return "TransBlue"


all_cmaps = get_colormaps()
good_cmap_keys = ['viridis', 'hsl', 'grays', 'RdBu', 'GrBu_d', 'hot']
filtered_cmaps = dict((k, all_cmaps[k]) for k in good_cmap_keys if k in all_cmaps)
opaque_cmaps = cycle(filtered_cmaps)
translucent_cmaps = cycle([TransFire(), TransGrays()])
opaque_cmap = next(opaque_cmaps)
translucent_cmap = next(translucent_cmaps)


class Test3DScene(unittest.TestCase):

    @unittest.skip
    def test_blank(self):
        canvas = scene.SceneCanvas(title='Blank Canvas')
        canvas.show()
        canvas.app.run()

    @unittest.skip
    def test_volume(self):
        canvas = scene.SceneCanvas(title='Blank Canvas')
        cView = canvas.central_widget.add_view()
        cAxis = scene.visuals.XYZAxis(parent=cView.scene)
        transform = STTransform(translate=(50., 50., 50.), scale=(100, 100, 100, 1))
        cAxis.transform = transform.as_matrix()
        description = scene.visuals.Text("Description",
                                         pos=(50, 20),
                                         anchor_x='left',
                                         bold=True,
                                         font_size=10,
                                         color='white',
                                         parent=cView)

        volume2 = scene.visuals.Volume(sox2_img,
                                       parent=cView.scene,
                                       threshold=1000,
                                       emulate_texture=False)
        volume2.cmap = TransRed()

        # volume = scene.visuals.Volume(syto16_img,
        #                               parent=cView.scene,
        #                               threshold=0.225,
        #                               emulate_texture=False)
        # volume.cmap = TransBlue()


        vz, vy, vx = [s / 2.0 for s in syto16_img.shape]
        cView.camera = scene.TurntableCamera(parent=cView.scene,
                                             fov=60.,
                                             center=(vx, vy, vz))
        canvas.show()
        canvas.app.run()

    @unittest.skip
    def test_multivolume(self):
        canvas = scene.SceneCanvas(title='Blank Canvas')
        cView = canvas.central_widget.add_view()

        reds = get_translucent_cmap(1, 0, 0)
        greens = get_translucent_cmap(0, 1, 0)
        blues = get_translucent_cmap(0, 0, 1)

        volumes = [
            (syto16_img, (0.1, 10), greens),
            (sox2_img, (0.1, 10), reds),
        ]

        volume1 = MultiVolume(volumes, parent=cView.scene, n_volume_max=2, relative_step_size=1.0)

        vz, vy, vx = [s / 2.0 for s in syto16_img.shape]
        cView.camera = scene.TurntableCamera(parent=cView.scene,
                                             fov=60.,
                                             center=(vx, vy, vz))
        canvas.show()
        canvas.app.run()

    @unittest.skip
    def test_rgba(self):
        canvas = scene.SceneCanvas(title='Blank Canvas')
        cView = canvas.central_widget.add_view()

        reds = np.array([1, 0, 0, 0.5])
        greens = np.array([0, 1, 0, 0.5])
        blues = np.array([0, 0, 1, 0.5])

        combined_data = np.zeros(syto16_img.shape + (4,))
        combined_data += syto16_img[:, :, :, np.newaxis] / 6. * blues
        combined_data += sox2_img[:, :, :, np.newaxis] / 6. * reds
        combined_data /= 5.

        volume1 = RGBAVolume(combined_data, parent=cView.scene)

        vz, vy, vx = [s / 2.0 for s in syto16_img.shape]
        cView.camera = scene.TurntableCamera(parent=cView.scene,
                                             fov=60.,
                                             center=(vx, vy, vz))
        canvas.show()
        canvas.app.run()


