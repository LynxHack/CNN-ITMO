from __future__ import division
from .helper import unittest, PillowLeakTestCase
import sys
from PIL import Image, features, ImageDraw, ImageFont


@unittest.skipIf(sys.platform.startswith('win32'), "requires Unix or macOS")
class TestTTypeFontLeak(PillowLeakTestCase):
    # fails at iteration 3 in master
    iterations = 10
    mem_limit = 4096  # k

    def _test_font(self, font):
        im = Image.new('RGB', (255, 255), 'white')
        draw = ImageDraw.ImageDraw(im)
        self._test_leak(lambda: draw.text((0, 0), "some text "*1024,  # ~10k
                                          font=font, fill="black"))

    @unittest.skipIf(not features.check('freetype2'),
                     "Test requires freetype2")
    def test_leak(self):
        ttype = ImageFont.truetype('Tests/fonts/FreeMono.ttf', 20)
        self._test_font(ttype)


class TestDefaultFontLeak(TestTTypeFontLeak):
    # fails at iteration 37 in master
    iterations = 100
    mem_limit = 1024  # k

    def test_leak(self):
        default_font = ImageFont.load_default()
        self._test_font(default_font)
