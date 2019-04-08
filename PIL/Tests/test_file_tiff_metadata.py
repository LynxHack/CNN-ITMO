import io
import struct

from .helper import PillowTestCase, hopper

from PIL import Image, TiffImagePlugin, TiffTags
from PIL.TiffImagePlugin import _limit_rational, IFDRational

tag_ids = {info.name: info.value for info in TiffTags.TAGS_V2.values()}


class TestFileTiffMetadata(PillowTestCase):

    def test_rt_metadata(self):
        """ Test writing arbitrary metadata into the tiff image directory
            Use case is ImageJ private tags, one numeric, one arbitrary
            data.  https://github.com/python-pillow/Pillow/issues/291
            """

        img = hopper()

        # Behaviour change: re #1416
        # Pre ifd rewrite, ImageJMetaData was being written as a string(2),
        # Post ifd rewrite, it's defined as arbitrary bytes(7). It should
        # roundtrip with the actual bytes, rather than stripped text
        # of the premerge tests.
        #
        # For text items, we still have to decode('ascii','replace') because
        # the tiff file format can't take 8 bit bytes in that field.

        basetextdata = "This is some arbitrary metadata for a text field"
        bindata = basetextdata.encode('ascii') + b" \xff"
        textdata = basetextdata + " " + chr(255)
        reloaded_textdata = basetextdata + " ?"
        floatdata = 12.345
        doubledata = 67.89
        info = TiffImagePlugin.ImageFileDirectory()

        ImageJMetaData = tag_ids['ImageJMetaData']
        ImageJMetaDataByteCounts = tag_ids['ImageJMetaDataByteCounts']
        ImageDescription = tag_ids['ImageDescription']

        info[ImageJMetaDataByteCounts] = len(bindata)
        info[ImageJMetaData] = bindata
        info[tag_ids['RollAngle']] = floatdata
        info.tagtype[tag_ids['RollAngle']] = 11
        info[tag_ids['YawAngle']] = doubledata
        info.tagtype[tag_ids['YawAngle']] = 12

        info[ImageDescription] = textdata

        f = self.tempfile("temp.tif")

        img.save(f, tiffinfo=info)

        loaded = Image.open(f)

        self.assertEqual(loaded.tag[ImageJMetaDataByteCounts], (len(bindata),))
        self.assertEqual(loaded.tag_v2[ImageJMetaDataByteCounts],
                         (len(bindata),))

        self.assertEqual(loaded.tag[ImageJMetaData], bindata)
        self.assertEqual(loaded.tag_v2[ImageJMetaData], bindata)

        self.assertEqual(loaded.tag[ImageDescription], (reloaded_textdata,))
        self.assertEqual(loaded.tag_v2[ImageDescription], reloaded_textdata)

        loaded_float = loaded.tag[tag_ids['RollAngle']][0]
        self.assertAlmostEqual(loaded_float, floatdata, places=5)
        loaded_double = loaded.tag[tag_ids['YawAngle']][0]
        self.assertAlmostEqual(loaded_double, doubledata)

        # check with 2 element ImageJMetaDataByteCounts, issue #2006

        info[ImageJMetaDataByteCounts] = (8, len(bindata) - 8)
        img.save(f, tiffinfo=info)
        loaded = Image.open(f)

        self.assertEqual(loaded.tag[ImageJMetaDataByteCounts],
                         (8, len(bindata) - 8))
        self.assertEqual(loaded.tag_v2[ImageJMetaDataByteCounts],
                         (8, len(bindata) - 8))

    def test_read_metadata(self):
        img = Image.open('Tests/images/hopper_g4.tif')

        self.assertEqual({'YResolution': IFDRational(4294967295, 113653537),
                          'PlanarConfiguration': 1,
                          'BitsPerSample': (1,),
                          'ImageLength': 128,
                          'Compression': 4,
                          'FillOrder': 1,
                          'RowsPerStrip': 128,
                          'ResolutionUnit': 3,
                          'PhotometricInterpretation': 0,
                          'PageNumber': (0, 1),
                          'XResolution': IFDRational(4294967295, 113653537),
                          'ImageWidth': 128,
                          'Orientation': 1,
                          'StripByteCounts': (1968,),
                          'SamplesPerPixel': 1,
                          'StripOffsets': (8,)
                          }, img.tag_v2.named())

        self.assertEqual({'YResolution': ((4294967295, 113653537),),
                          'PlanarConfiguration': (1,),
                          'BitsPerSample': (1,),
                          'ImageLength': (128,),
                          'Compression': (4,),
                          'FillOrder': (1,),
                          'RowsPerStrip': (128,),
                          'ResolutionUnit': (3,),
                          'PhotometricInterpretation': (0,),
                          'PageNumber': (0, 1),
                          'XResolution': ((4294967295, 113653537),),
                          'ImageWidth': (128,),
                          'Orientation': (1,),
                          'StripByteCounts': (1968,),
                          'SamplesPerPixel': (1,),
                          'StripOffsets': (8,)
                          }, img.tag.named())

    def test_write_metadata(self):
        """ Test metadata writing through the python code """
        img = Image.open('Tests/images/hopper.tif')

        f = self.tempfile('temp.tiff')
        img.save(f, tiffinfo=img.tag)

        loaded = Image.open(f)

        original = img.tag_v2.named()
        reloaded = loaded.tag_v2.named()

        for k, v in original.items():
            if isinstance(v, IFDRational):
                original[k] = IFDRational(*_limit_rational(v, 2**31))
            elif isinstance(v, tuple) and isinstance(v[0], IFDRational):
                original[k] = tuple(IFDRational(*_limit_rational(elt, 2**31))
                                    for elt in v)

        ignored = ['StripByteCounts', 'RowsPerStrip',
                   'PageNumber', 'StripOffsets']

        for tag, value in reloaded.items():
            if tag in ignored:
                continue
            if (isinstance(original[tag], tuple)
               and isinstance(original[tag][0], IFDRational)):
                # Need to compare element by element in the tuple,
                # not comparing tuples of object references
                self.assert_deep_equal(original[tag],
                                       value,
                                       "%s didn't roundtrip, %s, %s" %
                                       (tag, original[tag], value))
            else:
                self.assertEqual(original[tag],
                                 value,
                                 "%s didn't roundtrip, %s, %s" %
                                 (tag, original[tag], value))

        for tag, value in original.items():
            if tag not in ignored:
                self.assertEqual(
                    value, reloaded[tag], "%s didn't roundtrip" % tag)

    def test_no_duplicate_50741_tag(self):
        self.assertEqual(tag_ids['MakerNoteSafety'], 50741)
        self.assertEqual(tag_ids['BestQualityScale'], 50780)

    def test_empty_metadata(self):
        f = io.BytesIO(b'II*\x00\x08\x00\x00\x00')
        head = f.read(8)
        info = TiffImagePlugin.ImageFileDirectory(head)
        # Should not raise struct.error.
        self.assert_warning(UserWarning, info.load, f)

    def test_iccprofile(self):
        # https://github.com/python-pillow/Pillow/issues/1462
        im = Image.open('Tests/images/hopper.iccprofile.tif')
        out = self.tempfile('temp.tiff')

        im.save(out)
        reloaded = Image.open(out)
        self.assertNotIsInstance(im.info['icc_profile'], tuple)
        self.assertEqual(im.info['icc_profile'], reloaded.info['icc_profile'])

    def test_iccprofile_binary(self):
        # https://github.com/python-pillow/Pillow/issues/1526
        # We should be able to load this,
        # but probably won't be able to save it.

        im = Image.open('Tests/images/hopper.iccprofile_binary.tif')
        self.assertEqual(im.tag_v2.tagtype[34675], 1)
        self.assertTrue(im.info['icc_profile'])

    def test_iccprofile_save_png(self):
        im = Image.open('Tests/images/hopper.iccprofile.tif')
        outfile = self.tempfile('temp.png')
        im.save(outfile)

    def test_iccprofile_binary_save_png(self):
        im = Image.open('Tests/images/hopper.iccprofile_binary.tif')
        outfile = self.tempfile('temp.png')
        im.save(outfile)

    def test_exif_div_zero(self):
        im = hopper()
        info = TiffImagePlugin.ImageFileDirectory_v2()
        info[41988] = TiffImagePlugin.IFDRational(0, 0)

        out = self.tempfile('temp.tiff')
        im.save(out, tiffinfo=info, compression='raw')

        reloaded = Image.open(out)
        self.assertEqual(0, reloaded.tag_v2[41988].numerator)
        self.assertEqual(0, reloaded.tag_v2[41988].denominator)

    def test_expty_values(self):
        data = io.BytesIO(
            b'II*\x00\x08\x00\x00\x00\x03\x00\x1a\x01\x05\x00\x00\x00\x00\x00'
            b'\x00\x00\x00\x00\x1b\x01\x05\x00\x00\x00\x00\x00\x00\x00\x00\x00'
            b'\x98\x82\x02\x00\x07\x00\x00\x002\x00\x00\x00\x00\x00\x00\x00a '
            b'text\x00\x00')
        head = data.read(8)
        info = TiffImagePlugin.ImageFileDirectory_v2(head)
        info.load(data)
        # Should not raise ValueError.
        info = dict(info)
        self.assertIn(33432, info)

    def test_PhotoshopInfo(self):
        im = Image.open('Tests/images/issue_2278.tif')

        self.assertIsInstance(im.tag_v2[34377], bytes)
        out = self.tempfile('temp.tiff')
        im.save(out)
        reloaded = Image.open(out)
        self.assertIsInstance(reloaded.tag_v2[34377], bytes)

    def test_too_many_entries(self):
        ifd = TiffImagePlugin.ImageFileDirectory_v2()

        #    277: ("SamplesPerPixel", SHORT, 1),
        ifd._tagdata[277] = struct.pack('hh', 4, 4)
        ifd.tagtype[277] = TiffTags.SHORT

        # Should not raise ValueError.
        self.assert_warning(UserWarning, lambda: ifd[277])
