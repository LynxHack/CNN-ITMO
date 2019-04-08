from .helper import PillowTestCase, hopper

from PIL import Image
from PIL import ImageSequence
from PIL import SpiderImagePlugin

import tempfile

TEST_FILE = "Tests/images/hopper.spider"


class TestImageSpider(PillowTestCase):

    def test_sanity(self):
        im = Image.open(TEST_FILE)
        im.load()
        self.assertEqual(im.mode, "F")
        self.assertEqual(im.size, (128, 128))
        self.assertEqual(im.format, "SPIDER")

    def test_unclosed_file(self):
        def open():
            im = Image.open(TEST_FILE)
            im.load()
        self.assert_warning(None, open)

    def test_save(self):
        # Arrange
        temp = self.tempfile('temp.spider')
        im = hopper()

        # Act
        im.save(temp, "SPIDER")

        # Assert
        im2 = Image.open(temp)
        self.assertEqual(im2.mode, "F")
        self.assertEqual(im2.size, (128, 128))
        self.assertEqual(im2.format, "SPIDER")

    def test_tempfile(self):
        # Arrange
        im = hopper()

        # Act
        with tempfile.TemporaryFile() as fp:
            im.save(fp, "SPIDER")

            # Assert
            fp.seek(0)
            reloaded = Image.open(fp)
            self.assertEqual(reloaded.mode, "F")
            self.assertEqual(reloaded.size, (128, 128))
            self.assertEqual(reloaded.format, "SPIDER")

    def test_isSpiderImage(self):
        self.assertTrue(SpiderImagePlugin.isSpiderImage(TEST_FILE))

    def test_tell(self):
        # Arrange
        im = Image.open(TEST_FILE)

        # Act
        index = im.tell()

        # Assert
        self.assertEqual(index, 0)

    def test_n_frames(self):
        im = Image.open(TEST_FILE)
        self.assertEqual(im.n_frames, 1)
        self.assertFalse(im.is_animated)

    def test_loadImageSeries(self):
        # Arrange
        not_spider_file = "Tests/images/hopper.ppm"
        file_list = [TEST_FILE, not_spider_file, "path/not_found.ext"]

        # Act
        img_list = SpiderImagePlugin.loadImageSeries(file_list)

        # Assert
        self.assertEqual(len(img_list), 1)
        self.assertIsInstance(img_list[0], Image.Image)
        self.assertEqual(img_list[0].size, (128, 128))

    def test_loadImageSeries_no_input(self):
        # Arrange
        file_list = None

        # Act
        img_list = SpiderImagePlugin.loadImageSeries(file_list)

        # Assert
        self.assertIsNone(img_list)

    def test_isInt_not_a_number(self):
        # Arrange
        not_a_number = "a"

        # Act
        ret = SpiderImagePlugin.isInt(not_a_number)

        # Assert
        self.assertEqual(ret, 0)

    def test_invalid_file(self):
        invalid_file = "Tests/images/invalid.spider"

        self.assertRaises(IOError, Image.open, invalid_file)

    def test_nonstack_file(self):
        im = Image.open(TEST_FILE)

        self.assertRaises(EOFError, im.seek, 0)

    def test_nonstack_dos(self):
        im = Image.open(TEST_FILE)
        for i, frame in enumerate(ImageSequence.Iterator(im)):
            if i > 1:
                self.fail("Non-stack DOS file test failed")
