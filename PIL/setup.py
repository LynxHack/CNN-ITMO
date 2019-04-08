#!/usr/bin/env python
# > pyroma .
# ------------------------------
# Checking .
# Found Pillow
# ------------------------------
# Final rating: 10/10
# Your cheese is so fresh most people think it's a cream: Mascarpone
# ------------------------------
from __future__ import print_function

import os
import re
import struct
import subprocess
import sys
import warnings
from distutils import ccompiler, sysconfig
from distutils.command.build_ext import build_ext

from setuptools import Extension, setup

# monkey patch import hook. Even though flake8 says it's not used, it is.
# comment this out to disable multi threaded builds.
import mp_compile


if sys.platform == "win32" and sys.version_info >= (3, 8):
    warnings.warn(
        "Pillow does not yet support Python {}.{} and does not yet provide "
        "prebuilt Windows binaries. We do not recommend building from "
        "source on Windows.".format(sys.version_info.major,
                                    sys.version_info.minor),
        RuntimeWarning)


_IMAGING = ("decode", "encode", "map", "display", "outline", "path")

_LIB_IMAGING = (
    "Access", "AlphaComposite", "Resample", "Bands", "BcnDecode", "BitDecode",
    "Blend", "Chops", "ColorLUT", "Convert", "ConvertYCbCr", "Copy", "Crop",
    "Dib", "Draw", "Effects", "EpsEncode", "File", "Fill", "Filter",
    "FliDecode", "Geometry", "GetBBox", "GifDecode", "GifEncode", "HexDecode",
    "Histo", "JpegDecode", "JpegEncode", "Matrix", "ModeFilter",
    "Negative", "Offset", "Pack", "PackDecode", "Palette", "Paste", "Quant",
    "QuantOctree", "QuantHash", "QuantHeap", "PcdDecode", "PcxDecode",
    "PcxEncode", "Point", "RankFilter", "RawDecode", "RawEncode", "Storage",
    "SgiRleDecode", "SunRleDecode", "TgaRleDecode", "TgaRleEncode", "Unpack",
    "UnpackYCC", "UnsharpMask", "XbmDecode", "XbmEncode", "ZipDecode",
    "ZipEncode", "TiffDecode", "Jpeg2KDecode", "Jpeg2KEncode", "BoxBlur",
    "QuantPngQuant", "codec_fd")

DEBUG = False


class DependencyException(Exception):
    pass


class RequiredDependencyException(Exception):
    pass


PLATFORM_MINGW = 'mingw' in ccompiler.get_default_compiler()
PLATFORM_PYPY = hasattr(sys, 'pypy_version_info')


def _dbg(s, tp=None):
    if DEBUG:
        if tp:
            print(s % tp)
            return
        print(s)


def _find_library_dirs_ldconfig():
    # Based on ctypes.util from Python 2

    if sys.platform.startswith("linux") or sys.platform.startswith("gnu"):
        if struct.calcsize('l') == 4:
            machine = os.uname()[4] + '-32'
        else:
            machine = os.uname()[4] + '-64'
        mach_map = {
            'x86_64-64': 'libc6,x86-64',
            'ppc64-64': 'libc6,64bit',
            'sparc64-64': 'libc6,64bit',
            's390x-64': 'libc6,64bit',
            'ia64-64': 'libc6,IA-64',
            }
        abi_type = mach_map.get(machine, 'libc6')

        # Assuming GLIBC's ldconfig (with option -p)
        # Alpine Linux uses musl that can't print cache
        args = ['/sbin/ldconfig', '-p']
        expr = r'.*\(%s.*\) => (.*)' % abi_type
        env = dict(os.environ)
        env['LC_ALL'] = 'C'
        env['LANG'] = 'C'

    elif sys.platform.startswith("freebsd"):
        args = ['/sbin/ldconfig', '-r']
        expr = r'.* => (.*)'
        env = {}

    null = open(os.devnull, 'wb')
    try:
        with null:
            p = subprocess.Popen(args,
                                 stderr=null,
                                 stdout=subprocess.PIPE,
                                 env=env)
    except OSError:  # E.g. command not found
        return []
    [data, _] = p.communicate()
    if isinstance(data, bytes):
        data = data.decode()

    dirs = []
    for dll in re.findall(expr, data):
        dir = os.path.dirname(dll)
        if dir not in dirs:
            dirs.append(dir)
    return dirs


def _add_directory(path, subdir, where=None):
    if subdir is None:
        return
    subdir = os.path.realpath(subdir)
    if os.path.isdir(subdir) and subdir not in path:
        if where is None:
            _dbg('Appending path %s', subdir)
            path.append(subdir)
        else:
            _dbg('Inserting path %s', subdir)
            path.insert(where, subdir)
    elif subdir in path and where is not None:
        path.remove(subdir)
        path.insert(where, subdir)


def _find_include_file(self, include):
    for directory in self.compiler.include_dirs:
        _dbg('Checking for include file %s in %s', (include, directory))
        if os.path.isfile(os.path.join(directory, include)):
            _dbg('Found %s', include)
            return 1
    return 0


def _find_library_file(self, library):
    ret = self.compiler.find_library_file(self.compiler.library_dirs, library)
    if ret:
        _dbg('Found library %s at %s', (library, ret))
    else:
        _dbg("Couldn't find library %s in %s",
             (library, self.compiler.library_dirs))
    return ret


def _cmd_exists(cmd):
    return any(
        os.access(os.path.join(path, cmd), os.X_OK)
        for path in os.environ["PATH"].split(os.pathsep)
    )


def _read(file):
    with open(file, 'rb') as fp:
        return fp.read()


def get_version():
    version_file = 'src/PIL/_version.py'
    with open(version_file, 'r') as f:
        exec(compile(f.read(), version_file, 'exec'))
    return locals()['__version__']


NAME = 'Pillow'
PILLOW_VERSION = get_version()
JPEG_ROOT = None
JPEG2K_ROOT = None
ZLIB_ROOT = None
IMAGEQUANT_ROOT = None
TIFF_ROOT = None
FREETYPE_ROOT = None
LCMS_ROOT = None


def _pkg_config(name):
    try:
        command_libs = [
            'pkg-config',
            '--libs-only-L', name,
        ]
        command_cflags = [
            'pkg-config',
            '--cflags-only-I', name,
        ]
        if not DEBUG:
            command_libs.append('--silence-errors')
            command_cflags.append('--silence-errors')
        libs = (
            subprocess.check_output(command_libs)
            .decode("utf8")
            .strip()
            .replace("-L", "")
        )
        cflags = (
            subprocess.check_output(command_cflags)
            .decode("utf8")
            .strip()
            .replace("-I", "")
        )
        return (libs, cflags)
    except Exception:
        pass


class pil_build_ext(build_ext):
    class feature:
        features = ['zlib', 'jpeg', 'tiff', 'freetype', 'lcms', 'webp',
                    'webpmux', 'jpeg2000', 'imagequant']

        required = {'jpeg', 'zlib'}

        def __init__(self):
            for f in self.features:
                setattr(self, f, None)

        def require(self, feat):
            return feat in self.required

        def want(self, feat):
            return getattr(self, feat) is None

        def __iter__(self):
            for x in self.features:
                yield x

    feature = feature()

    user_options = build_ext.user_options + [
        ('disable-%s' % x, None, 'Disable support for %s' % x) for x in feature
    ] + [
        ('enable-%s' % x, None, 'Enable support for %s' % x) for x in feature
    ] + [
        ('disable-platform-guessing', None,
         'Disable platform guessing on Linux'),
        ('debug', None, 'Debug logging')
    ] + [('add-imaging-libs=', None, 'Add libs to _imaging build')]

    def initialize_options(self):
        self.disable_platform_guessing = None
        self.add_imaging_libs = ""
        build_ext.initialize_options(self)
        for x in self.feature:
            setattr(self, 'disable_%s' % x, None)
            setattr(self, 'enable_%s' % x, None)

    def finalize_options(self):
        build_ext.finalize_options(self)
        if self.debug:
            global DEBUG
            DEBUG = True
        if sys.version_info.major >= 3 and not self.parallel:
            # For Python 2.7, we monkeypatch distutils to have parallel
            # builds. If --parallel (or -j) wasn't specified, we want to
            # reproduce the same behavior as before, that is, auto-detect the
            # number of jobs.
            self.parallel = mp_compile.MAX_PROCS
        for x in self.feature:
            if getattr(self, 'disable_%s' % x):
                setattr(self.feature, x, False)
                self.feature.required.discard(x)
                _dbg('Disabling %s', x)
                if getattr(self, 'enable_%s' % x):
                    raise ValueError(
                        'Conflicting options: --enable-%s and --disable-%s' %
                        (x, x))
            if getattr(self, 'enable_%s' % x):
                _dbg('Requiring %s', x)
                self.feature.required.add(x)

    def build_extensions(self):

        library_dirs = []
        include_dirs = []

        _add_directory(include_dirs, "src/libImaging")

        pkg_config = None
        if _cmd_exists('pkg-config'):
            pkg_config = _pkg_config

        #
        # add configured kits
        for root_name, lib_name in dict(JPEG_ROOT="libjpeg",
                                        JPEG2K_ROOT="libopenjp2",
                                        TIFF_ROOT=("libtiff-5", "libtiff-4"),
                                        ZLIB_ROOT="zlib",
                                        FREETYPE_ROOT="freetype2",
                                        LCMS_ROOT="lcms2",
                                        IMAGEQUANT_ROOT="libimagequant"
                                        ).items():
            root = globals()[root_name]

            if root is None and root_name in os.environ:
                prefix = os.environ[root_name]
                root = (os.path.join(prefix, 'lib'), os.path.join(prefix, 'include'))

            if root is None and pkg_config:
                if isinstance(lib_name, tuple):
                    for lib_name2 in lib_name:
                        _dbg('Looking for `%s` using pkg-config.' % lib_name2)
                        root = pkg_config(lib_name2)
                        if root:
                            break
                else:
                    _dbg('Looking for `%s` using pkg-config.' % lib_name)
                    root = pkg_config(lib_name)

            if isinstance(root, tuple):
                lib_root, include_root = root
            else:
                lib_root = include_root = root

            _add_directory(library_dirs, lib_root)
            _add_directory(include_dirs, include_root)

        # respect CFLAGS/LDFLAGS
        for k in ('CFLAGS', 'LDFLAGS'):
            if k in os.environ:
                for match in re.finditer(r'-I([^\s]+)', os.environ[k]):
                    _add_directory(include_dirs, match.group(1))
                for match in re.finditer(r'-L([^\s]+)', os.environ[k]):
                    _add_directory(library_dirs, match.group(1))

        # include, rpath, if set as environment variables:
        for k in ('C_INCLUDE_PATH', 'CPATH', 'INCLUDE'):
            if k in os.environ:
                for d in os.environ[k].split(os.path.pathsep):
                    _add_directory(include_dirs, d)

        for k in ('LD_RUN_PATH', 'LIBRARY_PATH', 'LIB'):
            if k in os.environ:
                for d in os.environ[k].split(os.path.pathsep):
                    _add_directory(library_dirs, d)

        prefix = sysconfig.get_config_var("prefix")
        if prefix:
            _add_directory(library_dirs, os.path.join(prefix, "lib"))
            _add_directory(include_dirs, os.path.join(prefix, "include"))

        #
        # add platform directories

        if self.disable_platform_guessing:
            pass

        elif sys.platform == "cygwin":
            # pythonX.Y.dll.a is in the /usr/lib/pythonX.Y/config directory
            _add_directory(library_dirs,
                           os.path.join("/usr/lib", "python%s" %
                                        sys.version[:3], "config"))

        elif sys.platform == "darwin":
            # attempt to make sure we pick freetype2 over other versions
            _add_directory(include_dirs, "/sw/include/freetype2")
            _add_directory(include_dirs, "/sw/lib/freetype2/include")
            # fink installation directories
            _add_directory(library_dirs, "/sw/lib")
            _add_directory(include_dirs, "/sw/include")
            # darwin ports installation directories
            _add_directory(library_dirs, "/opt/local/lib")
            _add_directory(include_dirs, "/opt/local/include")

            # if Homebrew is installed, use its lib and include directories
            try:
                prefix = subprocess.check_output(['brew', '--prefix']).strip(
                ).decode('latin1')
            except Exception:
                # Homebrew not installed
                prefix = None

            ft_prefix = None

            if prefix:
                # add Homebrew's include and lib directories
                _add_directory(library_dirs, os.path.join(prefix, 'lib'))
                _add_directory(include_dirs, os.path.join(prefix, 'include'))
                ft_prefix = os.path.join(prefix, 'opt', 'freetype')

            if ft_prefix and os.path.isdir(ft_prefix):
                # freetype might not be linked into Homebrew's prefix
                _add_directory(library_dirs, os.path.join(ft_prefix, 'lib'))
                _add_directory(include_dirs,
                               os.path.join(ft_prefix, 'include'))
            else:
                # fall back to freetype from XQuartz if
                # Homebrew's freetype is missing
                _add_directory(library_dirs, "/usr/X11/lib")
                _add_directory(include_dirs, "/usr/X11/include")

        elif sys.platform.startswith("linux") or \
                sys.platform.startswith("gnu") or \
                sys.platform.startswith("freebsd"):
            for dirname in _find_library_dirs_ldconfig():
                _add_directory(library_dirs, dirname)
            if sys.platform.startswith("linux") and \
                    os.environ.get('ANDROID_ROOT', None):
                # termux support for android.
                # system libraries (zlib) are installed in /system/lib
                # headers are at $PREFIX/include
                # user libs are at $PREFIX/lib
                _add_directory(
                    library_dirs, os.path.join(os.environ["ANDROID_ROOT"], "lib")
                )

        elif sys.platform.startswith("netbsd"):
            _add_directory(library_dirs, "/usr/pkg/lib")
            _add_directory(include_dirs, "/usr/pkg/include")

        elif sys.platform.startswith("sunos5"):
            _add_directory(library_dirs, "/opt/local/lib")
            _add_directory(include_dirs, "/opt/local/include")

        # FIXME: check /opt/stuff directories here?

        # standard locations
        if not self.disable_platform_guessing:
            _add_directory(library_dirs, "/usr/local/lib")
            _add_directory(include_dirs, "/usr/local/include")

            _add_directory(library_dirs, "/usr/lib")
            _add_directory(include_dirs, "/usr/include")
            # alpine, at least
            _add_directory(library_dirs, "/lib")

        # on Windows, look for the OpenJPEG libraries in the location that
        # the official installer puts them
        if sys.platform == "win32":
            program_files = os.environ.get('ProgramFiles', '')
            best_version = (0, 0)
            best_path = None
            for name in os.listdir(program_files):
                if name.startswith('OpenJPEG '):
                    version = tuple(int(x) for x in
                                    name[9:].strip().split('.'))
                    if version > best_version:
                        best_version = version
                        best_path = os.path.join(program_files, name)

            if best_path:
                _dbg('Adding %s to search list', best_path)
                _add_directory(library_dirs, os.path.join(best_path, 'lib'))
                _add_directory(include_dirs,
                               os.path.join(best_path, 'include'))

        #
        # insert new dirs *before* default libs, to avoid conflicts
        # between Python PYD stub libs and real libraries

        self.compiler.library_dirs = library_dirs + self.compiler.library_dirs
        self.compiler.include_dirs = include_dirs + self.compiler.include_dirs

        #
        # look for available libraries

        feature = self.feature

        if feature.want('zlib'):
            _dbg('Looking for zlib')
            if _find_include_file(self, "zlib.h"):
                if _find_library_file(self, "z"):
                    feature.zlib = "z"
                elif (sys.platform == "win32" and
                      _find_library_file(self, "zlib")):
                    feature.zlib = "zlib"  # alternative name

        if feature.want('jpeg'):
            _dbg('Looking for jpeg')
            if _find_include_file(self, "jpeglib.h"):
                if _find_library_file(self, "jpeg"):
                    feature.jpeg = "jpeg"
                elif (sys.platform == "win32" and
                      _find_library_file(self, "libjpeg")):
                    feature.jpeg = "libjpeg"  # alternative name

        feature.openjpeg_version = None
        if feature.want('jpeg2000'):
            _dbg('Looking for jpeg2000')
            best_version = None
            best_path = None

            # Find the best version
            for directory in self.compiler.include_dirs:
                _dbg('Checking for openjpeg-#.# in %s', directory)
                try:
                    listdir = os.listdir(directory)
                except Exception:
                    # WindowsError, FileNotFoundError
                    continue
                for name in listdir:
                    if name.startswith('openjpeg-') and \
                        os.path.isfile(os.path.join(directory, name,
                                                    'openjpeg.h')):
                        _dbg('Found openjpeg.h in %s/%s', (directory, name))
                        version = tuple(int(x) for x in name[9:].split('.'))
                        if best_version is None or version > best_version:
                            best_version = version
                            best_path = os.path.join(directory, name)
                            _dbg('Best openjpeg version %s so far in %s',
                                 (best_version, best_path))

            if best_version and _find_library_file(self, 'openjp2'):
                # Add the directory to the include path so we can include
                # <openjpeg.h> rather than having to cope with the versioned
                # include path
                # FIXME (melvyn-sopacua):
                # At this point it's possible that best_path is already in
                # self.compiler.include_dirs. Should investigate how that is
                # possible.
                _add_directory(self.compiler.include_dirs, best_path, 0)
                feature.jpeg2000 = 'openjp2'
                feature.openjpeg_version = '.'.join(str(x) for x in
                                                    best_version)

        if feature.want('imagequant'):
            _dbg('Looking for imagequant')
            if _find_include_file(self, 'libimagequant.h'):
                if _find_library_file(self, "imagequant"):
                    feature.imagequant = "imagequant"
                elif _find_library_file(self, "libimagequant"):
                    feature.imagequant = "libimagequant"

        if feature.want('tiff'):
            _dbg('Looking for tiff')
            if _find_include_file(self, 'tiff.h'):
                if _find_library_file(self, "tiff"):
                    feature.tiff = "tiff"
                if (sys.platform in ["win32", "darwin"] and
                        _find_library_file(self, "libtiff")):
                    feature.tiff = "libtiff"

        if feature.want('freetype'):
            _dbg('Looking for freetype')
            if _find_library_file(self, "freetype"):
                # look for freetype2 include files
                freetype_version = 0
                for subdir in self.compiler.include_dirs:
                    _dbg('Checking for include file %s in %s',
                         ("ft2build.h", subdir))
                    if os.path.isfile(os.path.join(subdir, "ft2build.h")):
                        _dbg('Found %s in %s', ("ft2build.h", subdir))
                        freetype_version = 21
                        subdir = os.path.join(subdir, "freetype2")
                        break
                    subdir = os.path.join(subdir, "freetype2")
                    _dbg('Checking for include file %s in %s',
                         ("ft2build.h", subdir))
                    if os.path.isfile(os.path.join(subdir, "ft2build.h")):
                        _dbg('Found %s in %s', ("ft2build.h", subdir))
                        freetype_version = 21
                        break
                if freetype_version:
                    feature.freetype = "freetype"
                    if subdir:
                        _add_directory(self.compiler.include_dirs, subdir, 0)

        if feature.want('lcms'):
            _dbg('Looking for lcms')
            if _find_include_file(self, "lcms2.h"):
                if _find_library_file(self, "lcms2"):
                    feature.lcms = "lcms2"
                elif _find_library_file(self, "lcms2_static"):
                    # alternate Windows name.
                    feature.lcms = "lcms2_static"

        if feature.want('webp'):
            _dbg('Looking for webp')
            if (_find_include_file(self, "webp/encode.h") and
                    _find_include_file(self, "webp/decode.h")):
                # In Google's precompiled zip it is call "libwebp":
                if _find_library_file(self, "webp"):
                    feature.webp = "webp"
                elif _find_library_file(self, "libwebp"):
                    feature.webp = "libwebp"

        if feature.want('webpmux'):
            _dbg('Looking for webpmux')
            if (_find_include_file(self, "webp/mux.h") and
                    _find_include_file(self, "webp/demux.h")):
                if (_find_library_file(self, "webpmux") and
                        _find_library_file(self, "webpdemux")):
                    feature.webpmux = "webpmux"
                if (_find_library_file(self, "libwebpmux") and
                        _find_library_file(self, "libwebpdemux")):
                    feature.webpmux = "libwebpmux"

        for f in feature:
            if not getattr(feature, f) and feature.require(f):
                if f in ('jpeg', 'zlib'):
                    raise RequiredDependencyException(f)
                raise DependencyException(f)

        #
        # core library

        files = ["src/_imaging.c"]
        for src_file in _IMAGING:
            files.append("src/" + src_file + ".c")
        for src_file in _LIB_IMAGING:
            files.append(os.path.join("src/libImaging", src_file + ".c"))

        libs = self.add_imaging_libs.split()
        defs = []
        if feature.jpeg:
            libs.append(feature.jpeg)
            defs.append(("HAVE_LIBJPEG", None))
        if feature.jpeg2000:
            libs.append(feature.jpeg2000)
            defs.append(("HAVE_OPENJPEG", None))
            if sys.platform == "win32":
                defs.append(("OPJ_STATIC", None))
        if feature.zlib:
            libs.append(feature.zlib)
            defs.append(("HAVE_LIBZ", None))
        if feature.imagequant:
            libs.append(feature.imagequant)
            defs.append(("HAVE_LIBIMAGEQUANT", None))
        if feature.tiff:
            libs.append(feature.tiff)
            defs.append(("HAVE_LIBTIFF", None))
        if sys.platform == "win32":
            libs.extend(["kernel32", "user32", "gdi32"])
        if struct.unpack("h", "\0\1".encode('ascii'))[0] == 1:
            defs.append(("WORDS_BIGENDIAN", None))

        if sys.platform == "win32" and not (PLATFORM_PYPY or PLATFORM_MINGW):
            defs.append(("PILLOW_VERSION", '"\\"%s\\""' % PILLOW_VERSION))
        else:
            defs.append(("PILLOW_VERSION", '"%s"' % PILLOW_VERSION))

        exts = [(Extension("PIL._imaging",
                           files,
                           libraries=libs,
                           define_macros=defs))]

        #
        # additional libraries

        if feature.freetype:
            libs = ["freetype"]
            defs = []
            exts.append(Extension(
                "PIL._imagingft", ["src/_imagingft.c"], libraries=libs,
                define_macros=defs))

        if feature.lcms:
            extra = []
            if sys.platform == "win32":
                extra.extend(["user32", "gdi32"])
            exts.append(Extension("PIL._imagingcms",
                                  ["src/_imagingcms.c"],
                                  libraries=[feature.lcms] + extra))

        if feature.webp:
            libs = [feature.webp]
            defs = []

            if feature.webpmux:
                defs.append(("HAVE_WEBPMUX", None))
                libs.append(feature.webpmux)
                libs.append(feature.webpmux.replace('pmux', 'pdemux'))

            exts.append(Extension("PIL._webp",
                                  ["src/_webp.c"],
                                  libraries=libs,
                                  define_macros=defs))

        tk_libs = ['psapi'] if sys.platform == 'win32' else []
        exts.append(Extension("PIL._imagingtk",
                              ["src/_imagingtk.c", "src/Tk/tkImaging.c"],
                              include_dirs=['src/Tk'],
                              libraries=tk_libs))

        exts.append(Extension("PIL._imagingmath", ["src/_imagingmath.c"]))
        exts.append(Extension("PIL._imagingmorph", ["src/_imagingmorph.c"]))

        self.extensions[:] = exts

        build_ext.build_extensions(self)

        #
        # sanity checks

        self.summary_report(feature)

    def summary_report(self, feature):

        print("-" * 68)
        print("PIL SETUP SUMMARY")
        print("-" * 68)
        print("version      Pillow %s" % PILLOW_VERSION)
        v = sys.version.split("[")
        print("platform     %s %s" % (sys.platform, v[0].strip()))
        for v in v[1:]:
            print("             [%s" % v.strip())
        print("-" * 68)

        options = [
            (feature.jpeg, "JPEG"),
            (feature.jpeg2000, "OPENJPEG (JPEG2000)",
             feature.openjpeg_version),
            (feature.zlib, "ZLIB (PNG/ZIP)"),
            (feature.imagequant, "LIBIMAGEQUANT"),
            (feature.tiff, "LIBTIFF"),
            (feature.freetype, "FREETYPE2"),
            (feature.lcms, "LITTLECMS2"),
            (feature.webp, "WEBP"),
            (feature.webpmux, "WEBPMUX"),
        ]

        all = 1
        for option in options:
            if option[0]:
                version = ''
                if len(option) >= 3 and option[2]:
                    version = ' (%s)' % option[2]
                print("--- %s support available%s" % (option[1], version))
            else:
                print("*** %s support not available" % option[1])
                all = 0

        print("-" * 68)

        if not all:
            print("To add a missing option, make sure you have the required")
            print("library and headers.")
            print("See https://pillow.readthedocs.io/en/latest/installation."
                  "html#building-from-source")
            print("")

        print("To check the build, run the selftest.py script.")
        print("")


def debug_build():
    return hasattr(sys, 'gettotalrefcount')


needs_pytest = {'pytest', 'test', 'ptr'}.intersection(sys.argv)
pytest_runner = ['pytest-runner'] if needs_pytest else []

try:
    setup(name=NAME,
          version=PILLOW_VERSION,
          description='Python Imaging Library (Fork)',
          long_description=_read('README.rst').decode('utf-8'),
          author='Alex Clark (Fork Author)',
          author_email='aclark@aclark.net',
          url='http://python-pillow.org',
          classifiers=[
              "Development Status :: 6 - Mature",
              "License :: OSI Approved :: Historical Permission Notice and Disclaimer (HPND)",  # noqa: E501
              "Programming Language :: Python :: 2",
              "Programming Language :: Python :: 2.7",
              "Programming Language :: Python :: 3",
              "Programming Language :: Python :: 3.5",
              "Programming Language :: Python :: 3.6",
              "Programming Language :: Python :: 3.7",
              "Programming Language :: Python :: Implementation :: CPython",
              "Programming Language :: Python :: Implementation :: PyPy",
              "Topic :: Multimedia :: Graphics",
              "Topic :: Multimedia :: Graphics :: Capture :: Digital Camera",
              "Topic :: Multimedia :: Graphics :: Capture :: Screen Capture",
              "Topic :: Multimedia :: Graphics :: Graphics Conversion",
              "Topic :: Multimedia :: Graphics :: Viewers",
          ],
          python_requires=">=2.7, !=3.0.*, !=3.1.*, !=3.2.*, !=3.3.*, !=3.4.*",
          cmdclass={"build_ext": pil_build_ext},
          ext_modules=[Extension("PIL._imaging", ["_imaging.c"])],
          include_package_data=True,
          setup_requires=pytest_runner,
          tests_require=['pytest'],
          packages=["PIL"],
          package_dir={'': 'src'},
          keywords=["Imaging", ],
          zip_safe=not (debug_build() or PLATFORM_MINGW), )
except RequiredDependencyException as err:
    msg = """

The headers or library files could not be found for %s,
a required dependency when compiling Pillow from source.

Please see the install instructions at:
   https://pillow.readthedocs.io/en/latest/installation.html

""" % (str(err))
    sys.stderr.write(msg)
    raise RequiredDependencyException(msg)
except DependencyException as err:
    msg = """

The headers or library files could not be found for %s,
which was requested by the option flag --enable-%s

""" % (str(err), str(err))
    sys.stderr.write(msg)
    raise DependencyException(msg)
