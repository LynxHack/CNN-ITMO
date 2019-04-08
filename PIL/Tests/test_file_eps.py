from .helper import unittest, PillowTestCase, hopper

from PIL import Image, EpsImagePlugin
import io

HAS_GHOSTSCRIPT = EpsImagePlugin.has_ghostscript()

# Our two EPS test files (they are identical except for their bounding boxes)
file1 = "Tests/images/zero_bb.eps"
file2 = "Tests/images/non_zero_bb.eps"

# Due to palletization, we'll need to convert these to RGB after load
file1_compare = "Tests/images/zero_bb.png"
file1_compare_scale2 = "Tests/images/zero_bb_scale2.png"

file2_compare = "Tests/images/non_zero_bb.png"
file2_compare_scale2 = "Tests/images/non_zero_bb_scale2.png"

# EPS test files with binary preview
file3 = "Tests/images/binary_preview_map.eps"


class TestFileEps(PillowTestCase):

    @unittest.skipUnless(HAS_GHOSTSCRIPT, "Ghostscript not available")
    def test_sanity(self):
        # Regular scale
        image1 = Image.open(file1)
        image1.load()
        self.assertEqual(image1.mode, "RGB")
        self.assertEqual(image1.size, (460, 352))
        self.assertEqual(image1.format, "EPS")

        image2 = Image.open(file2)
        image2.load()
        self.assertEqual(image2.mode, "RGB")
        self.assertEqual(image2.size, (360, 252))
        self.assertEqual(image2.format, "EPS")

        # Double scale
        image1_scale2 = Image.open(file1)
        image1_scale2.load(scale=2)
        self.assertEqual(image1_scale2.mode, "RGB")
        self.assertEqual(image1_scale2.size, (920, 704))
        self.assertEqual(image1_scale2.format, "EPS")

        image2_scale2 = Image.open(file2)
        image2_scale2.load(scale=2)
        self.assertEqual(image2_scale2.mode, "RGB")
        self.assertEqual(image2_scale2.size, (720, 504))
        self.assertEqual(image2_scale2.format, "EPS")

    def test_invalid_file(self):
        invalid_file = "Tests/images/flower.jpg"

        self.assertRaises(SyntaxError,
                          EpsImagePlugin.EpsImageFile, invalid_file)

    @unittest.skipUnless(HAS_GHOSTSCRIPT, "Ghostscript not available")
    def test_cmyk(self):
        cmyk_image = Image.open("Tests/images/pil_sample_cmyk.eps")

        self.assertEqual(cmyk_image.mode, "CMYK")
        self.assertEqual(cmyk_image.size, (100, 100))
        self.assertEqual(cmyk_image.format, "EPS")

        cmyk_image.load()
        self.assertEqual(cmyk_image.mode, "RGB")

        if 'jpeg_decoder' in dir(Image.core):
            target = Image.open('Tests/images/pil_sample_rgb.jpg')
            self.assert_image_similar(cmyk_image, target, 10)

    @unittest.skipUnless(HAS_GHOSTSCRIPT, "Ghostscript not available")
    def test_showpage(self):
        # See https://github.com/python-pillow/Pillow/issues/2615
        plot_image = Image.open("Tests/images/reqd_showpage.eps")
        target = Image.open("Tests/images/reqd_showpage.png")

        # should not crash/hang
        plot_image.load()
        #  fonts could be slightly different
        self.assert_image_similar(plot_image, target, 6)

    @unittest.skipUnless(HAS_GHOSTSCRIPT, "Ghostscript not available")
    def test_file_object(self):
        # issue 479
        image1 = Image.open(file1)
        with open(self.tempfile('temp_file.eps'), 'wb') as fh:
            image1.save(fh, 'EPS')

    @unittest.skipUnless(HAS_GHOSTSCRIPT, "Ghostscript not available")
    def test_iobase_object(self):
        # issue 479
        image1 = Image.open(file1)
        with io.open(self.tempfile('temp_iobase.eps'), 'wb') as fh:
            image1.save(fh, 'EPS')

    @unittest.skipUnless(HAS_GHOSTSCRIPT, "Ghostscript not available")
    def test_bytesio_object(self):
        with open(file1, 'rb') as f:
            img_bytes = io.BytesIO(f.read())

        img = Image.open(img_bytes)
        img.load()

        image1_scale1_compare = Image.open(file1_compare).convert("RGB")
        image1_scale1_compare.load()
        self.assert_image_similar(img, image1_scale1_compare, 5)

    def test_image_mode_not_supported(self):
        im = hopper("RGBA")
        tmpfile = self.tempfile('temp.eps')
        self.assertRaises(ValueError, im.save, tmpfile)

    @unittest.skipUnless(HAS_GHOSTSCRIPT, "Ghostscript not available")
    def test_render_scale1(self):
        # We need png support for these render test
        codecs = dir(Image.core)
        if "zip_encoder" not in codecs or "zip_decoder" not in codecs:
            self.skipTest("zip/deflate support not available")

        # Zero bounding box
        image1_scale1 = Image.open(file1)
        image1_scale1.load()
        image1_scale1_compare = Image.open(file1_compare).convert("RGB")
        image1_scale1_compare.load()
        self.assert_image_similar(image1_scale1, image1_scale1_compare, 5)

        # Non-Zero bounding box
        image2_scale1 = Image.open(file2)
        image2_scale1.load()
        image2_scale1_compare = Image.open(file2_compare).convert("RGB")
        image2_scale1_compare.load()
        self.assert_image_similar(image2_scale1, image2_scale1_compare, 10)

    @unittest.skipUnless(HAS_GHOSTSCRIPT, "Ghostscript not available")
    def test_render_scale2(self):
        # We need png support for these render test
        codecs = dir(Image.core)
        if "zip_encoder" not in codecs or "zip_decoder" not in codecs:
            self.skipTest("zip/deflate support not available")

        # Zero bounding box
        image1_scale2 = Image.open(file1)
        image1_scale2.load(scale=2)
        image1_scale2_compare = Image.open(file1_compare_scale2).convert("RGB")
        image1_scale2_compare.load()
        self.assert_image_similar(image1_scale2, image1_scale2_compare, 5)

        # Non-Zero bounding box
        image2_scale2 = Image.open(file2)
        image2_scale2.load(scale=2)
        image2_scale2_compare = Image.open(file2_compare_scale2).convert("RGB")
        image2_scale2_compare.load()
        self.assert_image_similar(image2_scale2, image2_scale2_compare, 10)

    @unittest.skipUnless(HAS_GHOSTSCRIPT, "Ghostscript not available")
    def test_resize(self):
        # Arrange
        image1 = Image.open(file1)
        image2 = Image.open(file2)
        image3 = Image.open("Tests/images/illu10_preview.eps")
        new_size = (100, 100)

        # Act
        image1 = image1.resize(new_size)
        image2 = image2.resize(new_size)
        image3 = image3.resize(new_size)

        # Assert
        self.assertEqual(image1.size, new_size)
        self.assertEqual(image2.size, new_size)
        self.assertEqual(image3.size, new_size)

    @unittest.skipUnless(HAS_GHOSTSCRIPT, "Ghostscript not available")
    def test_thumbnail(self):
        # Issue #619
        # Arrange
        image1 = Image.open(file1)
        image2 = Image.open(file2)
        new_size = (100, 100)

        # Act
        image1.thumbnail(new_size)
        image2.thumbnail(new_size)

        # Assert
        self.assertEqual(max(image1.size), max(new_size))
        self.assertEqual(max(image2.size), max(new_size))

    def test_read_binary_preview(self):
        # Issue 302
        # open image with binary preview
        Image.open(file3)

    def _test_readline(self, t, ending):
        ending = "Failure with line ending: %s" % ("".join(
                                                   "%s" % ord(s)
                                                   for s in ending))
        self.assertEqual(t.readline().strip('\r\n'), 'something', ending)
        self.assertEqual(t.readline().strip('\r\n'), 'else', ending)
        self.assertEqual(t.readline().strip('\r\n'), 'baz', ending)
        self.assertEqual(t.readline().strip('\r\n'), 'bif', ending)

    def _test_readline_io_psfile(self, test_string, ending):
        f = io.BytesIO(test_string.encode('latin-1'))
        t = EpsImagePlugin.PSFile(f)
        self._test_readline(t, ending)

    def _test_readline_file_psfile(self, test_string, ending):
        f = self.tempfile('temp.txt')
        with open(f, 'wb') as w:
            w.write(test_string.encode('latin-1'))

        with open(f, 'rb') as r:
            t = EpsImagePlugin.PSFile(r)
            self._test_readline(t, ending)

    def test_readline(self):
        # check all the freaking line endings possible from the spec
        # test_string = u'something\r\nelse\n\rbaz\rbif\n'
        line_endings = ['\r\n', '\n', '\n\r', '\r']
        strings = ['something', 'else', 'baz', 'bif']

        for ending in line_endings:
            s = ending.join(strings)
            self._test_readline_io_psfile(s, ending)
            self._test_readline_file_psfile(s, ending)

    def test_open_eps(self):
        # https://github.com/python-pillow/Pillow/issues/1104
        # Arrange
        FILES = ["Tests/images/illu10_no_preview.eps",
                 "Tests/images/illu10_preview.eps",
                 "Tests/images/illuCS6_no_preview.eps",
                 "Tests/images/illuCS6_preview.eps"]

        # Act / Assert
        for filename in FILES:
            img = Image.open(filename)
            self.assertEqual(img.mode, "RGB")

    @unittest.skipUnless(HAS_GHOSTSCRIPT, "Ghostscript not available")
    def test_emptyline(self):
        # Test file includes an empty line in the header data
        emptyline_file = "Tests/images/zero_bb_emptyline.eps"

        image = Image.open(emptyline_file)
        image.load()
        self.assertEqual(image.mode, "RGB")
        self.assertEqual(image.size, (460, 352))
        self.assertEqual(image.format, "EPS")
