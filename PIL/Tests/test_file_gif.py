from .helper import unittest, PillowTestCase, hopper, netpbm_available

from PIL import Image, ImagePalette, GifImagePlugin

from io import BytesIO

try:
    from PIL import _webp
    HAVE_WEBP = True
except ImportError:
    HAVE_WEBP = False

codecs = dir(Image.core)

# sample gif stream
TEST_GIF = "Tests/images/hopper.gif"

with open(TEST_GIF, "rb") as f:
    data = f.read()


class TestFileGif(PillowTestCase):

    def setUp(self):
        if "gif_encoder" not in codecs or "gif_decoder" not in codecs:
            self.skipTest("gif support not available")  # can this happen?

    def test_sanity(self):
        im = Image.open(TEST_GIF)
        im.load()
        self.assertEqual(im.mode, "P")
        self.assertEqual(im.size, (128, 128))
        self.assertEqual(im.format, "GIF")
        self.assertEqual(im.info["version"], b"GIF89a")

    def test_unclosed_file(self):
        def open():
            im = Image.open(TEST_GIF)
            im.load()
        self.assert_warning(None, open)

    def test_invalid_file(self):
        invalid_file = "Tests/images/flower.jpg"

        self.assertRaises(SyntaxError,
                          GifImagePlugin.GifImageFile, invalid_file)

    def test_optimize(self):
        def test_grayscale(optimize):
            im = Image.new("L", (1, 1), 0)
            filename = BytesIO()
            im.save(filename, "GIF", optimize=optimize)
            return len(filename.getvalue())

        def test_bilevel(optimize):
            im = Image.new("1", (1, 1), 0)
            test_file = BytesIO()
            im.save(test_file, "GIF", optimize=optimize)
            return len(test_file.getvalue())

        self.assertEqual(test_grayscale(0), 800)
        self.assertEqual(test_grayscale(1), 38)
        self.assertEqual(test_bilevel(0), 800)
        self.assertEqual(test_bilevel(1), 800)

    def test_optimize_correctness(self):
        # 256 color Palette image, posterize to > 128 and < 128 levels
        # Size bigger and smaller than 512x512
        # Check the palette for number of colors allocated.
        # Check for correctness after conversion back to RGB
        def check(colors, size, expected_palette_length):
            # make an image with empty colors in the start of the palette range
            im = Image.frombytes('P', (colors, colors),
                                 bytes(bytearray(range(256-colors, 256))*colors))
            im = im.resize((size, size))
            outfile = BytesIO()
            im.save(outfile, 'GIF')
            outfile.seek(0)
            reloaded = Image.open(outfile)

            # check palette length
            palette_length = max(i+1 for i, v in enumerate(reloaded.histogram()) if v)
            self.assertEqual(expected_palette_length, palette_length)

            self.assert_image_equal(im.convert('RGB'), reloaded.convert('RGB'))

        # These do optimize the palette
        check(128, 511, 128)
        check(64, 511, 64)
        check(4, 511, 4)

        # These don't optimize the palette
        check(128, 513, 256)
        check(64, 513, 256)
        check(4, 513, 256)

        # other limits that don't optimize the palette
        check(129, 511, 256)
        check(255, 511, 256)
        check(256, 511, 256)

    def test_optimize_full_l(self):
        im = Image.frombytes("L", (16, 16), bytes(bytearray(range(256))))
        test_file = BytesIO()
        im.save(test_file, "GIF", optimize=True)
        self.assertEqual(im.mode, "L")

    def test_roundtrip(self):
        out = self.tempfile('temp.gif')
        im = hopper()
        im.save(out)
        reread = Image.open(out)

        self.assert_image_similar(reread.convert('RGB'), im, 50)

    def test_roundtrip2(self):
        # see https://github.com/python-pillow/Pillow/issues/403
        out = self.tempfile('temp.gif')
        im = Image.open(TEST_GIF)
        im2 = im.copy()
        im2.save(out)
        reread = Image.open(out)

        self.assert_image_similar(reread.convert('RGB'), hopper(), 50)

    def test_roundtrip_save_all(self):
        # Single frame image
        out = self.tempfile('temp.gif')
        im = hopper()
        im.save(out, save_all=True)
        reread = Image.open(out)

        self.assert_image_similar(reread.convert('RGB'), im, 50)

        # Multiframe image
        im = Image.open("Tests/images/dispose_bgnd.gif")

        out = self.tempfile('temp.gif')
        im.save(out, save_all=True)
        reread = Image.open(out)

        self.assertEqual(reread.n_frames, 5)

    def test_headers_saving_for_animated_gifs(self):
        important_headers = ['background', 'version', 'duration', 'loop']
        # Multiframe image
        im = Image.open("Tests/images/dispose_bgnd.gif")

        info = im.info.copy()

        out = self.tempfile('temp.gif')
        im.save(out, save_all=True)
        reread = Image.open(out)

        for header in important_headers:
            self.assertEqual(
                info[header],
                reread.info[header]
            )

    def test_palette_handling(self):
        # see https://github.com/python-pillow/Pillow/issues/513

        im = Image.open(TEST_GIF)
        im = im.convert('RGB')

        im = im.resize((100, 100), Image.LANCZOS)
        im2 = im.convert('P', palette=Image.ADAPTIVE, colors=256)

        f = self.tempfile('temp.gif')
        im2.save(f, optimize=True)

        reloaded = Image.open(f)

        self.assert_image_similar(im, reloaded.convert('RGB'), 10)

    def test_palette_434(self):
        # see https://github.com/python-pillow/Pillow/issues/434

        def roundtrip(im, *args, **kwargs):
            out = self.tempfile('temp.gif')
            im.copy().save(out, *args, **kwargs)
            reloaded = Image.open(out)

            return reloaded

        orig = "Tests/images/test.colors.gif"
        im = Image.open(orig)

        self.assert_image_similar(im, roundtrip(im), 1)
        self.assert_image_similar(im, roundtrip(im, optimize=True), 1)

        im = im.convert("RGB")
        # check automatic P conversion
        reloaded = roundtrip(im).convert('RGB')
        self.assert_image_equal(im, reloaded)

    @unittest.skipUnless(netpbm_available(), "netpbm not available")
    def test_save_netpbm_bmp_mode(self):
        img = Image.open(TEST_GIF).convert("RGB")

        tempfile = self.tempfile("temp.gif")
        GifImagePlugin._save_netpbm(img, 0, tempfile)
        self.assert_image_similar(img, Image.open(tempfile).convert("RGB"), 0)

    @unittest.skipUnless(netpbm_available(), "netpbm not available")
    def test_save_netpbm_l_mode(self):
        img = Image.open(TEST_GIF).convert("L")

        tempfile = self.tempfile("temp.gif")
        GifImagePlugin._save_netpbm(img, 0, tempfile)
        self.assert_image_similar(img, Image.open(tempfile).convert("L"), 0)

    def test_seek(self):
        img = Image.open("Tests/images/dispose_none.gif")
        framecount = 0
        try:
            while True:
                framecount += 1
                img.seek(img.tell() + 1)
        except EOFError:
            self.assertEqual(framecount, 5)

    def test_seek_info(self):
        im = Image.open("Tests/images/iss634.gif")
        info = im.info.copy()

        im.seek(1)
        im.seek(0)

        self.assertEqual(im.info, info)

    def test_seek_rewind(self):
        im = Image.open("Tests/images/iss634.gif")
        im.seek(2)
        im.seek(1)

        expected = Image.open("Tests/images/iss634.gif")
        expected.seek(1)
        self.assert_image_equal(im, expected)

    def test_n_frames(self):
        for path, n_frames in [
            [TEST_GIF, 1],
            ['Tests/images/iss634.gif', 42]
        ]:
            # Test is_animated before n_frames
            im = Image.open(path)
            self.assertEqual(im.is_animated, n_frames != 1)

            # Test is_animated after n_frames
            im = Image.open(path)
            self.assertEqual(im.n_frames, n_frames)
            self.assertEqual(im.is_animated, n_frames != 1)

    def test_eoferror(self):
        im = Image.open(TEST_GIF)
        n_frames = im.n_frames

        # Test seeking past the last frame
        self.assertRaises(EOFError, im.seek, n_frames)
        self.assertLess(im.tell(), n_frames)

        # Test that seeking to the last frame does not raise an error
        im.seek(n_frames-1)

    def test_dispose_none(self):
        img = Image.open("Tests/images/dispose_none.gif")
        try:
            while True:
                img.seek(img.tell() + 1)
                self.assertEqual(img.disposal_method, 1)
        except EOFError:
            pass

    def test_dispose_background(self):
        img = Image.open("Tests/images/dispose_bgnd.gif")
        try:
            while True:
                img.seek(img.tell() + 1)
                self.assertEqual(img.disposal_method, 2)
        except EOFError:
            pass

    def test_dispose_previous(self):
        img = Image.open("Tests/images/dispose_prev.gif")
        try:
            while True:
                img.seek(img.tell() + 1)
                self.assertEqual(img.disposal_method, 3)
        except EOFError:
            pass

    def test_save_dispose(self):
        out = self.tempfile('temp.gif')
        im_list = [
            Image.new('L', (100, 100), '#000'),
            Image.new('L', (100, 100), '#111'),
            Image.new('L', (100, 100), '#222'),
        ]
        for method in range(0, 4):
            im_list[0].save(
                out,
                save_all=True,
                append_images=im_list[1:],
                disposal=method
            )
            img = Image.open(out)
            for _ in range(2):
                img.seek(img.tell() + 1)
                self.assertEqual(img.disposal_method, method)

        # check per frame disposal
        im_list[0].save(
            out,
            save_all=True,
            append_images=im_list[1:],
            disposal=tuple(range(len(im_list)))
            )

        img = Image.open(out)

        for i in range(2):
            img.seek(img.tell() + 1)
            self.assertEqual(img.disposal_method, i+1)

    def test_iss634(self):
        img = Image.open("Tests/images/iss634.gif")
        # seek to the second frame
        img.seek(img.tell() + 1)
        # all transparent pixels should be replaced with the color from the
        # first frame
        self.assertEqual(img.histogram()[img.info['transparency']], 0)

    def test_duration(self):
        duration = 1000

        out = self.tempfile('temp.gif')
        im = Image.new('L', (100, 100), '#000')

        # Check that the argument has priority over the info settings
        im.info['duration'] = 100
        im.save(out, duration=duration)

        reread = Image.open(out)
        self.assertEqual(reread.info['duration'], duration)

    def test_multiple_duration(self):
        duration_list = [1000, 2000, 3000]

        out = self.tempfile('temp.gif')
        im_list = [
            Image.new('L', (100, 100), '#000'),
            Image.new('L', (100, 100), '#111'),
            Image.new('L', (100, 100), '#222')
        ]

        # duration as list
        im_list[0].save(
            out,
            save_all=True,
            append_images=im_list[1:],
            duration=duration_list
        )
        reread = Image.open(out)

        for duration in duration_list:
            self.assertEqual(reread.info['duration'], duration)
            try:
                reread.seek(reread.tell() + 1)
            except EOFError:
                pass

        # duration as tuple
        im_list[0].save(
            out,
            save_all=True,
            append_images=im_list[1:],
            duration=tuple(duration_list)
        )
        reread = Image.open(out)

        for duration in duration_list:
            self.assertEqual(reread.info['duration'], duration)
            try:
                reread.seek(reread.tell() + 1)
            except EOFError:
                pass

    def test_identical_frames(self):
        duration_list = [1000, 1500, 2000, 4000]

        out = self.tempfile('temp.gif')
        im_list = [
            Image.new('L', (100, 100), '#000'),
            Image.new('L', (100, 100), '#000'),
            Image.new('L', (100, 100), '#000'),
            Image.new('L', (100, 100), '#111')
        ]

        # duration as list
        im_list[0].save(
            out,
            save_all=True,
            append_images=im_list[1:],
            duration=duration_list
        )
        reread = Image.open(out)

        # Assert that the first three frames were combined
        self.assertEqual(reread.n_frames, 2)

        # Assert that the new duration is the total of the identical frames
        self.assertEqual(reread.info['duration'], 4500)

    def test_number_of_loops(self):
        number_of_loops = 2

        out = self.tempfile('temp.gif')
        im = Image.new('L', (100, 100), '#000')
        im.save(out, loop=number_of_loops)
        reread = Image.open(out)

        self.assertEqual(reread.info['loop'], number_of_loops)

    def test_background(self):
        out = self.tempfile('temp.gif')
        im = Image.new('L', (100, 100), '#000')
        im.info['background'] = 1
        im.save(out)
        reread = Image.open(out)

        self.assertEqual(reread.info['background'], im.info['background'])

        if HAVE_WEBP and _webp.HAVE_WEBPANIM:
            im = Image.open("Tests/images/hopper.webp")
            self.assertIsInstance(im.info['background'], tuple)
            im.save(out)

    def test_comment(self):
        im = Image.open(TEST_GIF)
        self.assertEqual(im.info['comment'],
                         b"File written by Adobe Photoshop\xa8 4.0")

        out = self.tempfile('temp.gif')
        im = Image.new('L', (100, 100), '#000')
        im.info['comment'] = b"Test comment text"
        im.save(out)
        reread = Image.open(out)

        self.assertEqual(reread.info['comment'], im.info['comment'])

    def test_comment_over_255(self):
        out = self.tempfile('temp.gif')
        im = Image.new('L', (100, 100), '#000')
        comment = b"Test comment text"
        while len(comment) < 256:
            comment += comment
        im.info['comment'] = comment
        im.save(out)
        reread = Image.open(out)

        self.assertEqual(reread.info['comment'], comment)

    def test_zero_comment_subblocks(self):
        im = Image.open('Tests/images/hopper_zero_comment_subblocks.gif')
        expected = Image.open(TEST_GIF)
        self.assert_image_equal(im, expected)

    def test_version(self):
        out = self.tempfile('temp.gif')

        def assertVersionAfterSave(im, version):
            im.save(out)
            reread = Image.open(out)
            self.assertEqual(reread.info["version"], version)

        # Test that GIF87a is used by default
        im = Image.new('L', (100, 100), '#000')
        assertVersionAfterSave(im, b"GIF87a")

        # Test setting the version to 89a
        im = Image.new('L', (100, 100), '#000')
        im.info["version"] = b"89a"
        assertVersionAfterSave(im, b"GIF89a")

        # Test that adding a GIF89a feature changes the version
        im.info["transparency"] = 1
        assertVersionAfterSave(im, b"GIF89a")

        # Test that a GIF87a image is also saved in that format
        im = Image.open("Tests/images/test.colors.gif")
        assertVersionAfterSave(im, b"GIF87a")

        # Test that a GIF89a image is also saved in that format
        im.info["version"] = b"GIF89a"
        assertVersionAfterSave(im, b"GIF87a")

    def test_append_images(self):
        out = self.tempfile('temp.gif')

        # Test appending single frame images
        im = Image.new('RGB', (100, 100), '#f00')
        ims = [Image.new('RGB', (100, 100), color) for color
               in ['#0f0', '#00f']]
        im.copy().save(out, save_all=True, append_images=ims)

        reread = Image.open(out)
        self.assertEqual(reread.n_frames, 3)

        # Tests appending using a generator
        def imGenerator(ims):
            for im in ims:
                yield im
        im.save(out, save_all=True, append_images=imGenerator(ims))

        reread = Image.open(out)
        self.assertEqual(reread.n_frames, 3)

        # Tests appending single and multiple frame images
        im = Image.open("Tests/images/dispose_none.gif")
        ims = [Image.open("Tests/images/dispose_prev.gif")]
        im.save(out, save_all=True, append_images=ims)

        reread = Image.open(out)
        self.assertEqual(reread.n_frames, 10)

    def test_transparent_optimize(self):
        # from issue #2195, if the transparent color is incorrectly
        # optimized out, gif loses transparency
        # Need a palette that isn't using the 0 color, and one
        # that's > 128 items where the transparent color is actually
        # the top palette entry to trigger the bug.

        data = bytes(bytearray(range(1, 254)))
        palette = ImagePalette.ImagePalette("RGB", list(range(256))*3)

        im = Image.new('L', (253, 1))
        im.frombytes(data)
        im.putpalette(palette)

        out = self.tempfile('temp.gif')
        im.save(out, transparency=253)
        reloaded = Image.open(out)

        self.assertEqual(reloaded.info['transparency'], 253)

    def test_rgb_transparency(self):
        out = self.tempfile('temp.gif')

        # Single frame
        im = Image.new('RGB', (1, 1))
        im.info['transparency'] = (255, 0, 0)
        self.assert_warning(UserWarning, im.save, out)

        reloaded = Image.open(out)
        self.assertNotIn('transparency', reloaded.info)

        # Multiple frames
        im = Image.new('RGB', (1, 1))
        im.info['transparency'] = b""
        ims = [Image.new('RGB', (1, 1))]
        self.assert_warning(UserWarning,
                            im.save, out, save_all=True, append_images=ims)

        reloaded = Image.open(out)
        self.assertNotIn('transparency', reloaded.info)

    def test_bbox(self):
        out = self.tempfile('temp.gif')

        im = Image.new('RGB', (100, 100), '#fff')
        ims = [Image.new("RGB", (100, 100), '#000')]
        im.save(out, save_all=True, append_images=ims)

        reread = Image.open(out)
        self.assertEqual(reread.n_frames, 2)

    def test_palette_save_L(self):
        # generate an L mode image with a separate palette

        im = hopper('P')
        im_l = Image.frombytes('L', im.size, im.tobytes())
        palette = bytes(bytearray(im.getpalette()))

        out = self.tempfile('temp.gif')
        im_l.save(out, palette=palette)

        reloaded = Image.open(out)

        self.assert_image_equal(reloaded.convert('RGB'), im.convert('RGB'))

    def test_palette_save_P(self):
        # pass in a different palette, then construct what the image
        # would look like.
        # Forcing a non-straight grayscale palette.

        im = hopper('P')
        palette = bytes(bytearray([255-i//3 for i in range(768)]))

        out = self.tempfile('temp.gif')
        im.save(out, palette=palette)

        reloaded = Image.open(out)
        im.putpalette(palette)
        self.assert_image_equal(reloaded, im)

    def test_palette_save_ImagePalette(self):
        # pass in a different palette, as an ImagePalette.ImagePalette
        # effectively the same as test_palette_save_P

        im = hopper('P')
        palette = ImagePalette.ImagePalette('RGB', list(range(256))[::-1]*3)

        out = self.tempfile('temp.gif')
        im.save(out, palette=palette)

        reloaded = Image.open(out)
        im.putpalette(palette)
        self.assert_image_equal(reloaded, im)

    def test_save_I(self):
        # Test saving something that would trigger the auto-convert to 'L'

        im = hopper('I')

        out = self.tempfile('temp.gif')
        im.save(out)

        reloaded = Image.open(out)
        self.assert_image_equal(reloaded.convert('L'), im.convert('L'))

    def test_getdata(self):
        # test getheader/getdata against legacy values
        # Create a 'P' image with holes in the palette
        im = Image._wedge().resize((16, 16))
        im.putpalette(ImagePalette.ImagePalette('RGB'))
        im.info = {'background': 0}

        passed_palette = bytes(bytearray([255-i//3 for i in range(768)]))

        GifImagePlugin._FORCE_OPTIMIZE = True
        try:
            h = GifImagePlugin.getheader(im, passed_palette)
            d = GifImagePlugin.getdata(im)

            import pickle
            # Enable to get target values on pre-refactor version
            # with open('Tests/images/gif_header_data.pkl', 'wb') as f:
            #    pickle.dump((h, d), f, 1)
            with open('Tests/images/gif_header_data.pkl', 'rb') as f:
                (h_target, d_target) = pickle.load(f)

            self.assertEqual(h, h_target)
            self.assertEqual(d, d_target)
        finally:
            GifImagePlugin._FORCE_OPTIMIZE = False

    def test_lzw_bits(self):
        # see https://github.com/python-pillow/Pillow/issues/2811
        im = Image.open('Tests/images/issue_2811.gif')

        self.assertEqual(im.tile[0][3][0], 11)  # LZW bits
        # codec error prepatch
        im.load()
