import unittest

import numpy as np
from cognitive import stim_generator as sg
from PIL import Image


class StimGeneratorTest(unittest.TestCase):
    def testRenderStatisObj(self):
        for _ in range(5):
            canvas = np.zeros([224, 224, 3], dtype=np.uint8)
            space = sg.Space()
            object = sg.Object([sg.Shape(0), space.sample()])
            object.epoch = (0, 1)
            object = object.to_static()[0]
            sg.render_static_obj(canvas, object, 224)
            Image.fromarray(canvas).show()

    def testGetShapeNet(self):
        img = sg.get_shapenet_object(3, (2, 2))
        self.assertEqual(img.size, (2, 2))


if __name__ == '__main__':
    unittest.main()
