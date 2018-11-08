# This file implements a RGBAVolumeVisual class that can be used to show
# multiple volumes simultaneously. It is derived from the original VolumeVisual
# class in vispy.visuals.volume, which is releaed under a BSD license included
# here:
#
# ===========================================================================
# Vispy is licensed under the terms of the (new) BSD license:
#
# Copyright (c) 2015, authors of Vispy
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# * Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in the
#   documentation and/or other materials provided with the distribution.
# * Neither the name of Vispy Development Team nor the names of its
#   contributors may be used to endorse or promote products
#   derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
# IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
# TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
# OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ===========================================================================
#
# This modified version is released under the BSD license given in the LICENSE
# file in this repository.


from vispy.gloo import Texture3D, TextureEmulated3D, VertexBuffer, IndexBuffer
from vispy.visuals.volume import VolumeVisual, Visual
from vispy.scene.visuals import create_visual_node

import numpy as np

from .rgba_volume_shaders import VERT_SHADER, FRAG_SHADER


class RGBAVolumeVisual(VolumeVisual):
    """
    Carry out additive volume rendering using an RGBA cube instead of a 3d cube
    of values and a colormap.

    Parameters
    ----------
    data : np.ndarray
        A 4-d array with dimensions (z, y, x, 4) where the last dimension
        corresponds to RGBA. The data should be normalized from 0 to 1 in each
        channel.
    relative_step_size : float
        The relative step size to step through the volume. Default 0.8.
        Increase to e.g. 1.5 to increase performance, at the cost of
        quality.
    emulate_texture : bool
        Use 2D textures to emulate a 3D texture. OpenGL ES 2.0 compatible,
        but has lower performance on desktop platforms.
    """

    def __init__(self, data, relative_step_size=0.8,
                 emulate_texture=False):

        # Choose texture class
        tex_cls = TextureEmulated3D if emulate_texture else Texture3D

        # Storage of information of volume
        self._need_vertex_update = True

        # Create OpenGL program
        Visual.__init__(self, vcode=VERT_SHADER, fcode=FRAG_SHADER)

        # Create gloo objects
        self._vertices = VertexBuffer()
        self._texcoord = VertexBuffer(
            np.array([
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [1, 1, 0],
                [0, 0, 1],
                [1, 0, 1],
                [0, 1, 1],
                [1, 1, 1],
            ], dtype=np.float32))

        # Set up RGBA texture
        self.texture = tex_cls((10, 10, 10, 4), interpolation='linear',
                                                wrapping='clamp_to_edge')
        self.texture.set_data(data.astype(np.float32))
        self.shared_program['u_volumetex'] = self.texture

        self._vol_shape = data.shape[:-1]
        self.shared_program['u_shape'] = self._vol_shape[::-1]

        self.shared_program['a_position'] = self._vertices
        self.shared_program['a_texcoord'] = self._texcoord

        self._draw_mode = 'triangle_strip'
        self._index_buffer = IndexBuffer()

        self.shared_program.frag['sampler_type'] = self.texture.glsl_sampler_type
        self.shared_program.frag['sample'] = self.texture.glsl_sample

        # Only show back faces of cuboid. This is required because if we are
        # inside the volume, then the front faces are outside of the clipping
        # box and will not be drawn.
        self.set_gl_state('translucent', cull_face=False)

        self.relative_step_size = relative_step_size
        self.freeze()


RGBAVolume = create_visual_node(RGBAVolumeVisual)
