"""
Helper functions.
"""
from __future__ import print_function
import sys
import tempfile
import os
import unittest

from PIL import Image, ImageMath
from PIL._util import py3

import logging
logger = logging.getLogger(__name__)


HAS_UPLOADER = False

if os.environ.get('SHOW_ERRORS', None):
    # local img.show for errors.
    HAS_UPLOADER = True

    class test_image_results:
        @classmethod
        def upload(self, a, b):
            a.show()
            b.show()
else:
    try:
        import test_image_results
        HAS_UPLOADER = True
    except ImportError:
        pass


def convert_to_comparable(a, b):
    new_a, new_b = a, b
    if a.mode == 'P':
        new_a = Image.new('L', a.size)
        new_b = Image.new('L', b.size)
        new_a.putdata(a.getdata())
        new_b.putdata(b.getdata())
    elif a.mode == 'I;16':
        new_a = a.convert('I')
        new_b = b.convert('I')
    return new_a, new_b


class PillowTestCase(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        unittest.TestCase.__init__(self, *args, **kwargs)
        # holds last result object passed to run method:
        self.currentResult = None

    def run(self, result=None):
        self.currentResult = result  # remember result for use later
        unittest.TestCase.run(self, result)  # call superclass run method

    def delete_tempfile(self, path):
        try:
            ok = self.currentResult.wasSuccessful()
        except AttributeError:  # for pytest
            ok = True

        if ok:
            # only clean out tempfiles if test passed
            try:
                os.remove(path)
            except OSError:
                pass  # report?
        else:
            print("=== orphaned temp file: %s" % path)

    def assert_deep_equal(self, a, b, msg=None):
        try:
            self.assertEqual(
                len(a), len(b),
                msg or "got length %s, expected %s" % (len(a), len(b)))
            self.assertTrue(
                all(x == y for x, y in zip(a, b)),
                msg or "got %s, expected %s" % (a, b))
        except Exception:
            self.assertEqual(a, b, msg)

    def assert_image(self, im, mode, size, msg=None):
        if mode is not None:
            self.assertEqual(
                im.mode, mode,
                msg or "got mode %r, expected %r" % (im.mode, mode))

        if size is not None:
            self.assertEqual(
                im.size, size,
                msg or "got size %r, expected %r" % (im.size, size))

    def assert_image_equal(self, a, b, msg=None):
        self.assertEqual(
            a.mode, b.mode,
            msg or "got mode %r, expected %r" % (a.mode, b.mode))
        self.assertEqual(
            a.size, b.size,
            msg or "got size %r, expected %r" % (a.size, b.size))
        if a.tobytes() != b.tobytes():
            if HAS_UPLOADER:
                try:
                    url = test_image_results.upload(a, b)
                    logger.error("Url for test images: %s" % url)
                except Exception:
                    pass

            self.fail(msg or "got different content")

    def assert_image_equal_tofile(self, a, filename, msg=None, mode=None):
        with Image.open(filename) as img:
            if mode:
                img = img.convert(mode)
            self.assert_image_equal(a, img, msg)

    def assert_image_similar(self, a, b, epsilon, msg=None):
        epsilon = float(epsilon)
        self.assertEqual(
            a.mode, b.mode,
            msg or "got mode %r, expected %r" % (a.mode, b.mode))
        self.assertEqual(
            a.size, b.size,
            msg or "got size %r, expected %r" % (a.size, b.size))

        a, b = convert_to_comparable(a, b)

        diff = 0
        for ach, bch in zip(a.split(), b.split()):
            chdiff = ImageMath.eval("abs(a - b)", a=ach, b=bch).convert('L')
            diff += sum(i * num for i, num in enumerate(chdiff.histogram()))

        ave_diff = float(diff)/(a.size[0]*a.size[1])
        try:
            self.assertGreaterEqual(
                epsilon, ave_diff,
                (msg or '') +
                " average pixel value difference %.4f > epsilon %.4f" % (
                    ave_diff, epsilon))
        except Exception as e:
            if HAS_UPLOADER:
                try:
                    url = test_image_results.upload(a, b)
                    logger.error("Url for test images: %s" % url)
                except Exception:
                    pass
            raise e

    def assert_image_similar_tofile(self, a, filename, epsilon, msg=None,
                                    mode=None):
        with Image.open(filename) as img:
            if mode:
                img = img.convert(mode)
            self.assert_image_similar(a, img, epsilon, msg)

    def assert_warning(self, warn_class, func, *args, **kwargs):
        import warnings

        with warnings.catch_warnings(record=True) as w:
            # Cause all warnings to always be triggered.
            warnings.simplefilter("always")

            # Hopefully trigger a warning.
            result = func(*args, **kwargs)

            # Verify some things.
            if warn_class is None:
                self.assertEqual(len(w), 0,
                                 "Expected no warnings, got %s" %
                                 [v.category for v in w])
            else:
                self.assertGreaterEqual(len(w), 1)
                found = False
                for v in w:
                    if issubclass(v.category, warn_class):
                        found = True
                        break
                self.assertTrue(found)
        return result

    def assert_all_same(self, items, msg=None):
        self.assertEqual(items.count(items[0]), len(items), msg)

    def assert_not_all_same(self, items, msg=None):
        self.assertNotEqual(items.count(items[0]), len(items), msg)

    def assert_tuple_approx_equal(self, actuals, targets, threshold, msg):
        """Tests if actuals has values within threshold from targets"""

        value = True
        for i, target in enumerate(targets):
            value *= (target - threshold <= actuals[i] <= target + threshold)

        self.assertTrue(value,
                        msg + ': ' + repr(actuals) + ' != ' + repr(targets))

    def skipKnownBadTest(self, msg=None, platform=None,
                         travis=None, interpreter=None):
        # Skip if platform/travis matches, and
        # PILLOW_RUN_KNOWN_BAD is not true in the environment.
        if os.environ.get('PILLOW_RUN_KNOWN_BAD', False):
            print(os.environ.get('PILLOW_RUN_KNOWN_BAD', False))
            return

        skip = True
        if platform is not None:
            skip = sys.platform.startswith(platform)
        if travis is not None:
            skip = skip and (travis == bool(os.environ.get('TRAVIS', False)))
        if interpreter is not None:
            skip = skip and (interpreter == 'pypy' and
                             hasattr(sys, 'pypy_version_info'))
        if skip:
            self.skipTest(msg or "Known Bad Test")

    def tempfile(self, template):
        assert template[:5] in ("temp.", "temp_")
        fd, path = tempfile.mkstemp(template[4:], template[:4])
        os.close(fd)

        self.addCleanup(self.delete_tempfile, path)
        return path

    def open_withImagemagick(self, f):
        if not imagemagick_available():
            raise IOError()

        outfile = self.tempfile("temp.png")
        if command_succeeds([IMCONVERT, f, outfile]):
            return Image.open(outfile)
        raise IOError()


@unittest.skipIf(sys.platform.startswith('win32'), "requires Unix or macOS")
class PillowLeakTestCase(PillowTestCase):
    # requires unix/macOS
    iterations = 100  # count
    mem_limit = 512  # k

    def _get_mem_usage(self):
        """
        Gets the RUSAGE memory usage, returns in K. Encapsulates the difference
        between macOS and Linux rss reporting

        :returns: memory usage in kilobytes
        """

        from resource import getrusage, RUSAGE_SELF
        mem = getrusage(RUSAGE_SELF).ru_maxrss
        if sys.platform == 'darwin':
            # man 2 getrusage:
            #     ru_maxrss
            # This is the maximum resident set size utilized (in bytes).
            return mem / 1024  # Kb
        else:
            # linux
            # man 2 getrusage
            #        ru_maxrss (since Linux 2.6.32)
            #  This is the maximum resident set size used (in kilobytes).
            return mem  # Kb

    def _test_leak(self, core):
        start_mem = self._get_mem_usage()
        for cycle in range(self.iterations):
            core()
            mem = (self._get_mem_usage() - start_mem)
            msg = 'memory usage limit exceeded in iteration %d' % cycle
            self.assertLess(mem, self.mem_limit, msg)


# helpers

if not py3:
    # Remove DeprecationWarning in Python 3
    PillowTestCase.assertRaisesRegex = PillowTestCase.assertRaisesRegexp
    PillowTestCase.assertRegex = PillowTestCase.assertRegexpMatches


def fromstring(data):
    from io import BytesIO
    return Image.open(BytesIO(data))


def tostring(im, string_format, **options):
    from io import BytesIO
    out = BytesIO()
    im.save(out, string_format, **options)
    return out.getvalue()


def hopper(mode=None, cache={}):
    if mode is None:
        # Always return fresh not-yet-loaded version of image.
        # Operations on not-yet-loaded images is separate class of errors
        # what we should catch.
        return Image.open("Tests/images/hopper.ppm")
    # Use caching to reduce reading from disk but so an original copy is
    # returned each time and the cached image isn't modified by tests
    # (for fast, isolated, repeatable tests).
    im = cache.get(mode)
    if im is None:
        if mode == "F":
            im = hopper("L").convert(mode)
        elif mode[:4] == "I;16":
            im = hopper("I").convert(mode)
        else:
            im = hopper().convert(mode)
        cache[mode] = im
    return im.copy()


def command_succeeds(cmd):
    """
    Runs the command, which must be a list of strings. Returns True if the
    command succeeds, or False if an OSError was raised by subprocess.Popen.
    """
    import subprocess
    with open(os.devnull, 'wb') as f:
        try:
            subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT)
        except OSError:
            return False
    return True


def djpeg_available():
    return command_succeeds(['djpeg', '-version'])


def cjpeg_available():
    return command_succeeds(['cjpeg', '-version'])


def netpbm_available():
    return (command_succeeds(["ppmquant", "--version"]) and
            command_succeeds(["ppmtogif", "--version"]))


def imagemagick_available():
    return IMCONVERT and command_succeeds([IMCONVERT, '-version'])


def on_appveyor():
    return 'APPVEYOR' in os.environ


if sys.platform == 'win32':
    IMCONVERT = os.environ.get('MAGICK_HOME', '')
    if IMCONVERT:
        IMCONVERT = os.path.join(IMCONVERT, 'convert.exe')
else:
    IMCONVERT = 'convert'


def distro():
    if os.path.exists('/etc/os-release'):
        with open('/etc/os-release', 'r') as f:
            for line in f:
                if 'ID=' in line:
                    return line.strip().split('=')[1]


class cached_property(object):
    def __init__(self, func):
        self.func = func

    def __get__(self, instance, cls=None):
        result = instance.__dict__[self.func.__name__] = self.func(instance)
        return result
