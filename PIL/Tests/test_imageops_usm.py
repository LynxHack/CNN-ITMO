from .helper import PillowTestCase

from PIL import Image
from PIL import ImageFilter

im = Image.open("Tests/images/hopper.ppm")
snakes = Image.open("Tests/images/color_snakes.png")


class TestImageOpsUsm(PillowTestCase):

    def test_filter_api(self):

        test_filter = ImageFilter.GaussianBlur(2.0)
        i = im.filter(test_filter)
        self.assertEqual(i.mode, "RGB")
        self.assertEqual(i.size, (128, 128))

        test_filter = ImageFilter.UnsharpMask(2.0, 125, 8)
        i = im.filter(test_filter)
        self.assertEqual(i.mode, "RGB")
        self.assertEqual(i.size, (128, 128))

    def test_usm_formats(self):

        usm = ImageFilter.UnsharpMask
        self.assertRaises(ValueError, im.convert("1").filter, usm)
        im.convert("L").filter(usm)
        self.assertRaises(ValueError, im.convert("I").filter, usm)
        self.assertRaises(ValueError, im.convert("F").filter, usm)
        im.convert("RGB").filter(usm)
        im.convert("RGBA").filter(usm)
        im.convert("CMYK").filter(usm)
        self.assertRaises(ValueError, im.convert("YCbCr").filter, usm)

    def test_blur_formats(self):

        blur = ImageFilter.GaussianBlur
        self.assertRaises(ValueError, im.convert("1").filter, blur)
        blur(im.convert("L"))
        self.assertRaises(ValueError, im.convert("I").filter, blur)
        self.assertRaises(ValueError, im.convert("F").filter, blur)
        im.convert("RGB").filter(blur)
        im.convert("RGBA").filter(blur)
        im.convert("CMYK").filter(blur)
        self.assertRaises(ValueError, im.convert("YCbCr").filter, blur)

    def test_usm_accuracy(self):

        src = snakes.convert('RGB')
        i = src.filter(ImageFilter.UnsharpMask(5, 1024, 0))
        # Image should not be changed because it have only 0 and 255 levels.
        self.assertEqual(i.tobytes(), src.tobytes())

    def test_blur_accuracy(self):

        i = snakes.filter(ImageFilter.GaussianBlur(.4))
        # These pixels surrounded with pixels with 255 intensity.
        # They must be very close to 255.
        for x, y, c in [(1, 0, 1), (2, 0, 1), (7, 8, 1), (8, 8, 1), (2, 9, 1),
                        (7, 3, 0), (8, 3, 0), (5, 8, 0), (5, 9, 0), (1, 3, 0),
                        (4, 3, 2), (4, 2, 2)]:
            self.assertGreaterEqual(i.im.getpixel((x, y))[c], 250)
        # Fuzzy match.

        def gp(x, y):
            return i.im.getpixel((x, y))
        self.assertTrue(236 <= gp(7, 4)[0] <= 239)
        self.assertTrue(236 <= gp(7, 5)[2] <= 239)
        self.assertTrue(236 <= gp(7, 6)[2] <= 239)
        self.assertTrue(236 <= gp(7, 7)[1] <= 239)
        self.assertTrue(236 <= gp(8, 4)[0] <= 239)
        self.assertTrue(236 <= gp(8, 5)[2] <= 239)
        self.assertTrue(236 <= gp(8, 6)[2] <= 239)
        self.assertTrue(236 <= gp(8, 7)[1] <= 239)
