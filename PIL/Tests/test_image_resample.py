from __future__ import division, print_function

from contextlib import contextmanager

from .helper import unittest, PillowTestCase, hopper
from PIL import Image, ImageDraw


class TestImagingResampleVulnerability(PillowTestCase):
    # see https://github.com/python-pillow/Pillow/issues/1710
    def test_overflow(self):
        im = hopper('L')
        xsize = 0x100000008 // 4
        ysize = 1000  # unimportant
        with self.assertRaises(MemoryError):
            # any resampling filter will do here
            im.im.resize((xsize, ysize), Image.BILINEAR)

    def test_invalid_size(self):
        im = hopper()

        # Should not crash
        im.resize((100, 100))

        with self.assertRaises(ValueError):
            im.resize((-100, 100))

        with self.assertRaises(ValueError):
            im.resize((100, -100))

    def test_modify_after_resizing(self):
        im = hopper('RGB')
        # get copy with same size
        copy = im.resize(im.size)
        # some in-place operation
        copy.paste('black', (0, 0, im.width // 2, im.height // 2))
        # image should be different
        self.assertNotEqual(im.tobytes(), copy.tobytes())


class TestImagingCoreResampleAccuracy(PillowTestCase):
    def make_case(self, mode, size, color):
        """Makes a sample image with two dark and two bright squares.
        For example:
        e0 e0 1f 1f
        e0 e0 1f 1f
        1f 1f e0 e0
        1f 1f e0 e0
        """
        case = Image.new('L', size, 255 - color)
        rectangle = ImageDraw.Draw(case).rectangle
        rectangle((0, 0, size[0] // 2 - 1, size[1] // 2 - 1), color)
        rectangle((size[0] // 2, size[1] // 2, size[0], size[1]), color)

        return Image.merge(mode, [case] * len(mode))

    def make_sample(self, data, size):
        """Restores a sample image from given data string which contains
        hex-encoded pixels from the top left fourth of a sample.
        """
        data = data.replace(' ', '')
        sample = Image.new('L', size)
        s_px = sample.load()
        w, h = size[0] // 2, size[1] // 2
        for y in range(h):
            for x in range(w):
                val = int(data[(y * w + x) * 2:(y * w + x + 1) * 2], 16)
                s_px[x, y] = val
                s_px[size[0] - x - 1, size[1] - y - 1] = val
                s_px[x, size[1] - y - 1] = 255 - val
                s_px[size[0] - x - 1, y] = 255 - val
        return sample

    def check_case(self, case, sample):
        s_px = sample.load()
        c_px = case.load()
        for y in range(case.size[1]):
            for x in range(case.size[0]):
                if c_px[x, y] != s_px[x, y]:
                    message = '\nHave: \n{}\n\nExpected: \n{}'.format(
                        self.serialize_image(case),
                        self.serialize_image(sample),
                    )
                    self.assertEqual(s_px[x, y], c_px[x, y], message)

    def serialize_image(self, image):
        s_px = image.load()
        return '\n'.join(
            ' '.join(
                '{:02x}'.format(s_px[x, y])
                for x in range(image.size[0])
            )
            for y in range(image.size[1])
        )

    def test_reduce_box(self):
        for mode in ['RGBX', 'RGB', 'La', 'L']:
            case = self.make_case(mode, (8, 8), 0xe1)
            case = case.resize((4, 4), Image.BOX)
            data = ('e1 e1'
                    'e1 e1')
            for channel in case.split():
                self.check_case(channel, self.make_sample(data, (4, 4)))

    def test_reduce_bilinear(self):
        for mode in ['RGBX', 'RGB', 'La', 'L']:
            case = self.make_case(mode, (8, 8), 0xe1)
            case = case.resize((4, 4), Image.BILINEAR)
            data = ('e1 c9'
                    'c9 b7')
            for channel in case.split():
                self.check_case(channel, self.make_sample(data, (4, 4)))

    def test_reduce_hamming(self):
        for mode in ['RGBX', 'RGB', 'La', 'L']:
            case = self.make_case(mode, (8, 8), 0xe1)
            case = case.resize((4, 4), Image.HAMMING)
            data = ('e1 da'
                    'da d3')
            for channel in case.split():
                self.check_case(channel, self.make_sample(data, (4, 4)))

    def test_reduce_bicubic(self):
        for mode in ['RGBX', 'RGB', 'La', 'L']:
            case = self.make_case(mode, (12, 12), 0xe1)
            case = case.resize((6, 6), Image.BICUBIC)
            data = ('e1 e3 d4'
                    'e3 e5 d6'
                    'd4 d6 c9')
            for channel in case.split():
                self.check_case(channel, self.make_sample(data, (6, 6)))

    def test_reduce_lanczos(self):
        for mode in ['RGBX', 'RGB', 'La', 'L']:
            case = self.make_case(mode, (16, 16), 0xe1)
            case = case.resize((8, 8), Image.LANCZOS)
            data = ('e1 e0 e4 d7'
                    'e0 df e3 d6'
                    'e4 e3 e7 da'
                    'd7 d6 d9 ce')
            for channel in case.split():
                self.check_case(channel, self.make_sample(data, (8, 8)))

    def test_enlarge_box(self):
        for mode in ['RGBX', 'RGB', 'La', 'L']:
            case = self.make_case(mode, (2, 2), 0xe1)
            case = case.resize((4, 4), Image.BOX)
            data = ('e1 e1'
                    'e1 e1')
            for channel in case.split():
                self.check_case(channel, self.make_sample(data, (4, 4)))

    def test_enlarge_bilinear(self):
        for mode in ['RGBX', 'RGB', 'La', 'L']:
            case = self.make_case(mode, (2, 2), 0xe1)
            case = case.resize((4, 4), Image.BILINEAR)
            data = ('e1 b0'
                    'b0 98')
            for channel in case.split():
                self.check_case(channel, self.make_sample(data, (4, 4)))

    def test_enlarge_hamming(self):
        for mode in ['RGBX', 'RGB', 'La', 'L']:
            case = self.make_case(mode, (2, 2), 0xe1)
            case = case.resize((4, 4), Image.HAMMING)
            data = ('e1 d2'
                    'd2 c5')
            for channel in case.split():
                self.check_case(channel, self.make_sample(data, (4, 4)))

    def test_enlarge_bicubic(self):
        for mode in ['RGBX', 'RGB', 'La', 'L']:
            case = self.make_case(mode, (4, 4), 0xe1)
            case = case.resize((8, 8), Image.BICUBIC)
            data = ('e1 e5 ee b9'
                    'e5 e9 f3 bc'
                    'ee f3 fd c1'
                    'b9 bc c1 a2')
            for channel in case.split():
                self.check_case(channel, self.make_sample(data, (8, 8)))

    def test_enlarge_lanczos(self):
        for mode in ['RGBX', 'RGB', 'La', 'L']:
            case = self.make_case(mode, (6, 6), 0xe1)
            case = case.resize((12, 12), Image.LANCZOS)
            data = ('e1 e0 db ed f5 b8'
                    'e0 df da ec f3 b7'
                    'db db d6 e7 ee b5'
                    'ed ec e6 fb ff bf'
                    'f5 f4 ee ff ff c4'
                    'b8 b7 b4 bf c4 a0')
            for channel in case.split():
                self.check_case(channel, self.make_sample(data, (12, 12)))


class CoreResampleConsistencyTest(PillowTestCase):
    def make_case(self, mode, fill):
        im = Image.new(mode, (512, 9), fill)
        return im.resize((9, 512), Image.LANCZOS), im.load()[0, 0]

    def run_case(self, case):
        channel, color = case
        px = channel.load()
        for x in range(channel.size[0]):
            for y in range(channel.size[1]):
                if px[x, y] != color:
                    message = "{} != {} for pixel {}".format(
                        px[x, y], color, (x, y))
                    self.assertEqual(px[x, y], color, message)

    def test_8u(self):
        im, color = self.make_case('RGB', (0, 64, 255))
        r, g, b = im.split()
        self.run_case((r, color[0]))
        self.run_case((g, color[1]))
        self.run_case((b, color[2]))
        self.run_case(self.make_case('L', 12))

    def test_32i(self):
        self.run_case(self.make_case('I', 12))
        self.run_case(self.make_case('I', 0x7fffffff))
        self.run_case(self.make_case('I', -12))
        self.run_case(self.make_case('I', -1 << 31))

    def test_32f(self):
        self.run_case(self.make_case('F', 1))
        self.run_case(self.make_case('F', 3.40282306074e+38))
        self.run_case(self.make_case('F', 1.175494e-38))
        self.run_case(self.make_case('F', 1.192093e-07))


class CoreResampleAlphaCorrectTest(PillowTestCase):
    def make_levels_case(self, mode):
        i = Image.new(mode, (256, 16))
        px = i.load()
        for y in range(i.size[1]):
            for x in range(i.size[0]):
                pix = [x] * len(mode)
                pix[-1] = 255 - y * 16
                px[x, y] = tuple(pix)
        return i

    def run_levels_case(self, i):
        px = i.load()
        for y in range(i.size[1]):
            used_colors = {px[x, y][0] for x in range(i.size[0])}
            self.assertEqual(256, len(used_colors),
                             'All colors should present in resized image. '
                             'Only {} on {} line.'.format(len(used_colors), y))

    @unittest.skip("current implementation isn't precise enough")
    def test_levels_rgba(self):
        case = self.make_levels_case('RGBA')
        self.run_levels_case(case.resize((512, 32), Image.BOX))
        self.run_levels_case(case.resize((512, 32), Image.BILINEAR))
        self.run_levels_case(case.resize((512, 32), Image.HAMMING))
        self.run_levels_case(case.resize((512, 32), Image.BICUBIC))
        self.run_levels_case(case.resize((512, 32), Image.LANCZOS))

    @unittest.skip("current implementation isn't precise enough")
    def test_levels_la(self):
        case = self.make_levels_case('LA')
        self.run_levels_case(case.resize((512, 32), Image.BOX))
        self.run_levels_case(case.resize((512, 32), Image.BILINEAR))
        self.run_levels_case(case.resize((512, 32), Image.HAMMING))
        self.run_levels_case(case.resize((512, 32), Image.BICUBIC))
        self.run_levels_case(case.resize((512, 32), Image.LANCZOS))

    def make_dirty_case(self, mode, clean_pixel, dirty_pixel):
        i = Image.new(mode, (64, 64), dirty_pixel)
        px = i.load()
        xdiv4 = i.size[0] // 4
        ydiv4 = i.size[1] // 4
        for y in range(ydiv4 * 2):
            for x in range(xdiv4 * 2):
                px[x + xdiv4, y + ydiv4] = clean_pixel
        return i

    def run_dirty_case(self, i, clean_pixel):
        px = i.load()
        for y in range(i.size[1]):
            for x in range(i.size[0]):
                if px[x, y][-1] != 0 and px[x, y][:-1] != clean_pixel:
                    message = 'pixel at ({}, {}) is differ:\n{}\n{}'\
                        .format(x, y, px[x, y], clean_pixel)
                    self.assertEqual(px[x, y][:3], clean_pixel, message)

    def test_dirty_pixels_rgba(self):
        case = self.make_dirty_case('RGBA', (255, 255, 0, 128), (0, 0, 255, 0))
        self.run_dirty_case(case.resize((20, 20), Image.BOX), (255, 255, 0))
        self.run_dirty_case(case.resize((20, 20), Image.BILINEAR),
                            (255, 255, 0))
        self.run_dirty_case(case.resize((20, 20), Image.HAMMING),
                            (255, 255, 0))
        self.run_dirty_case(case.resize((20, 20), Image.BICUBIC),
                            (255, 255, 0))
        self.run_dirty_case(case.resize((20, 20), Image.LANCZOS),
                            (255, 255, 0))

    def test_dirty_pixels_la(self):
        case = self.make_dirty_case('LA', (255, 128), (0, 0))
        self.run_dirty_case(case.resize((20, 20), Image.BOX), (255,))
        self.run_dirty_case(case.resize((20, 20), Image.BILINEAR), (255,))
        self.run_dirty_case(case.resize((20, 20), Image.HAMMING), (255,))
        self.run_dirty_case(case.resize((20, 20), Image.BICUBIC), (255,))
        self.run_dirty_case(case.resize((20, 20), Image.LANCZOS), (255,))


class CoreResamplePassesTest(PillowTestCase):
    @contextmanager
    def count(self, diff):
        count = Image.core.get_stats()['new_count']
        yield
        self.assertEqual(Image.core.get_stats()['new_count'] - count, diff)

    def test_horizontal(self):
        im = hopper('L')
        with self.count(1):
            im.resize((im.size[0] - 10, im.size[1]), Image.BILINEAR)

    def test_vertical(self):
        im = hopper('L')
        with self.count(1):
            im.resize((im.size[0], im.size[1] - 10), Image.BILINEAR)

    def test_both(self):
        im = hopper('L')
        with self.count(2):
            im.resize((im.size[0] - 10, im.size[1] - 10), Image.BILINEAR)

    def test_box_horizontal(self):
        im = hopper('L')
        box = (20, 0, im.size[0] - 20, im.size[1])
        with self.count(1):
            # the same size, but different box
            with_box = im.resize(im.size, Image.BILINEAR, box)
        with self.count(2):
            cropped = im.crop(box).resize(im.size, Image.BILINEAR)
        self.assert_image_similar(with_box, cropped, 0.1)

    def test_box_vertical(self):
        im = hopper('L')
        box = (0, 20, im.size[0], im.size[1] - 20)
        with self.count(1):
            # the same size, but different box
            with_box = im.resize(im.size, Image.BILINEAR, box)
        with self.count(2):
            cropped = im.crop(box).resize(im.size, Image.BILINEAR)
        self.assert_image_similar(with_box, cropped, 0.1)


class CoreResampleCoefficientsTest(PillowTestCase):
    def test_reduce(self):
        test_color = 254

        for size in range(400000, 400010, 2):
            i = Image.new('L', (size, 1), 0)
            draw = ImageDraw.Draw(i)
            draw.rectangle((0, 0, i.size[0] // 2 - 1, 0), test_color)

            px = i.resize((5, i.size[1]), Image.BICUBIC).load()
            if px[2, 0] != test_color // 2:
                self.assertEqual(test_color // 2, px[2, 0])

    def test_nonzero_coefficients(self):
        # regression test for the wrong coefficients calculation
        # due to bug https://github.com/python-pillow/Pillow/issues/2161
        im = Image.new('RGBA', (1280, 1280), (0x20, 0x40, 0x60, 0xff))
        histogram = im.resize((256, 256), Image.BICUBIC).histogram()

        # first channel
        self.assertEqual(histogram[0x100 * 0 + 0x20], 0x10000)
        # second channel
        self.assertEqual(histogram[0x100 * 1 + 0x40], 0x10000)
        # third channel
        self.assertEqual(histogram[0x100 * 2 + 0x60], 0x10000)
        # fourth channel
        self.assertEqual(histogram[0x100 * 3 + 0xff], 0x10000)


class CoreResampleBoxTest(PillowTestCase):
    def test_wrong_arguments(self):
        im = hopper()
        for resample in (Image.NEAREST, Image.BOX, Image.BILINEAR,
                         Image.HAMMING, Image.BICUBIC, Image.LANCZOS):
            im.resize((32, 32), resample, (0, 0, im.width, im.height))
            im.resize((32, 32), resample, (20, 20, im.width, im.height))
            im.resize((32, 32), resample, (20, 20, 20, 100))
            im.resize((32, 32), resample, (20, 20, 100, 20))

            with self.assertRaisesRegex(TypeError,
                                        "must be sequence of length 4"):
                im.resize((32, 32), resample, (im.width, im.height))

            with self.assertRaisesRegex(ValueError, "can't be negative"):
                im.resize((32, 32), resample, (-20, 20, 100, 100))
            with self.assertRaisesRegex(ValueError, "can't be negative"):
                im.resize((32, 32), resample, (20, -20, 100, 100))

            with self.assertRaisesRegex(ValueError, "can't be empty"):
                im.resize((32, 32), resample, (20.1, 20, 20, 100))
            with self.assertRaisesRegex(ValueError, "can't be empty"):
                im.resize((32, 32), resample, (20, 20.1, 100, 20))
            with self.assertRaisesRegex(ValueError, "can't be empty"):
                im.resize((32, 32), resample, (20.1, 20.1, 20, 20))

            with self.assertRaisesRegex(ValueError, "can't exceed"):
                im.resize((32, 32), resample, (0, 0, im.width + 1, im.height))
            with self.assertRaisesRegex(ValueError, "can't exceed"):
                im.resize((32, 32), resample, (0, 0, im.width, im.height + 1))

    def resize_tiled(self, im, dst_size, xtiles, ytiles):
        def split_range(size, tiles):
            scale = size / tiles
            for i in range(tiles):
                yield (int(round(scale * i)), int(round(scale * (i + 1))))

        tiled = Image.new(im.mode, dst_size)
        scale = (im.size[0] / tiled.size[0], im.size[1] / tiled.size[1])

        for y0, y1 in split_range(dst_size[1], ytiles):
            for x0, x1 in split_range(dst_size[0], xtiles):
                box = (x0 * scale[0], y0 * scale[1],
                       x1 * scale[0], y1 * scale[1])
                tile = im.resize((x1 - x0, y1 - y0), Image.BICUBIC, box)
                tiled.paste(tile, (x0, y0))
        return tiled

    def test_tiles(self):
        im = Image.open("Tests/images/flower.jpg")
        self.assertEqual(im.size, (480, 360))
        dst_size = (251, 188)
        reference = im.resize(dst_size, Image.BICUBIC)

        for tiles in [(1, 1), (3, 3), (9, 7), (100, 100)]:
            tiled = self.resize_tiled(im, dst_size, *tiles)
            self.assert_image_similar(reference, tiled, 0.01)

    def test_subsample(self):
        # This test shows advantages of the subpixel resizing
        # after supersampling (e.g. during JPEG decoding).
        im = Image.open("Tests/images/flower.jpg")
        self.assertEqual(im.size, (480, 360))
        dst_size = (48, 36)
        # Reference is cropped image resized to destination
        reference = im.crop((0, 0, 473, 353)).resize(dst_size, Image.BICUBIC)
        # Image.BOX emulates supersampling (480 / 8 = 60, 360 / 8 = 45)
        supersampled = im.resize((60, 45), Image.BOX)

        with_box = supersampled.resize(dst_size, Image.BICUBIC,
                                       (0, 0, 59.125, 44.125))
        without_box = supersampled.resize(dst_size, Image.BICUBIC)

        # error with box should be much smaller than without
        self.assert_image_similar(reference, with_box, 6)
        with self.assertRaisesRegex(AssertionError, r"difference 29\."):
            self.assert_image_similar(reference, without_box, 5)

    def test_formats(self):
        for resample in [Image.NEAREST, Image.BILINEAR]:
            for mode in ['RGB', 'L', 'RGBA', 'LA', 'I', '']:
                im = hopper(mode)
                box = (20, 20, im.size[0] - 20, im.size[1] - 20)
                with_box = im.resize((32, 32), resample, box)
                cropped = im.crop(box).resize((32, 32), resample)
                self.assert_image_similar(cropped, with_box, 0.4)

    def test_passthrough(self):
        # When no resize is required
        im = hopper()

        for size, box in [
            ((40, 50), (0, 0, 40, 50)),
            ((40, 50), (0, 10, 40, 60)),
            ((40, 50), (10, 0, 50, 50)),
            ((40, 50), (10, 20, 50, 70)),
        ]:
            try:
                res = im.resize(size, Image.LANCZOS, box)
                self.assertEqual(res.size, size)
                self.assert_image_equal(res, im.crop(box))
            except AssertionError:
                print('>>>', size, box)
                raise

    def test_no_passthrough(self):
        # When resize is required
        im = hopper()

        for size, box in [
            ((40, 50), (0.4, 0.4, 40.4, 50.4)),
            ((40, 50), (0.4, 10.4, 40.4, 60.4)),
            ((40, 50), (10.4, 0.4, 50.4, 50.4)),
            ((40, 50), (10.4, 20.4, 50.4, 70.4)),
        ]:
            try:
                res = im.resize(size, Image.LANCZOS, box)
                self.assertEqual(res.size, size)
                with self.assertRaisesRegex(AssertionError, r"difference \d"):
                    # check that the difference at least that much
                    self.assert_image_similar(res, im.crop(box), 20)
            except AssertionError:
                print('>>>', size, box)
                raise

    def test_skip_horizontal(self):
        # Can skip resize for one dimension
        im = hopper()

        for flt in [Image.NEAREST, Image.BICUBIC]:
            for size, box in [
                ((40, 50), (0, 0, 40, 90)),
                ((40, 50), (0, 20, 40, 90)),
                ((40, 50), (10, 0, 50, 90)),
                ((40, 50), (10, 20, 50, 90)),
            ]:
                try:
                    res = im.resize(size, flt, box)
                    self.assertEqual(res.size, size)
                    # Borders should be slightly different
                    self.assert_image_similar(
                        res, im.crop(box).resize(size, flt), 0.4)
                except AssertionError:
                    print('>>>', size, box, flt)
                    raise

    def test_skip_vertical(self):
        # Can skip resize for one dimension
        im = hopper()

        for flt in [Image.NEAREST, Image.BICUBIC]:
            for size, box in [
                ((40, 50), (0, 0, 90, 50)),
                ((40, 50), (20, 0, 90, 50)),
                ((40, 50), (0, 10, 90, 60)),
                ((40, 50), (20, 10, 90, 60)),
            ]:
                try:
                    res = im.resize(size, flt, box)
                    self.assertEqual(res.size, size)
                    # Borders should be slightly different
                    self.assert_image_similar(
                        res, im.crop(box).resize(size, flt), 0.4)
                except AssertionError:
                    print('>>>', size, box, flt)
                    raise
