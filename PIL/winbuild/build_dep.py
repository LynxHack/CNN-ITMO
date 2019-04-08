from unzip import unzip
from untar import untar
import os

from fetch import fetch
from config import (compilers, all_compilers, compiler_from_env, bit_from_env,
                    libs)
from build import vc_setup


def _relpath(*args):
    return os.path.join(os.getcwd(), *args)


build_dir = _relpath('build')
inc_dir = _relpath('depends')


def check_sig(filename, signame):
    # UNDONE -- need gpg
    return filename


def mkdirs():
    try:
        os.mkdir(build_dir)
    except OSError:
        pass
    try:
        os.mkdir(inc_dir)
    except OSError:
        pass
    for compiler in all_compilers():
        try:
            os.mkdir(os.path.join(inc_dir, compiler['inc_dir']))
        except OSError:
            pass


def extract(src, dest):
    if '.zip' in src:
        return unzip(src, dest)
    if '.tar.gz' in src or '.tgz' in src:
        return untar(src, dest)


def extract_libs():
    for name, lib in libs.items():
        filename = lib['filename']
        if not os.path.exists(filename):
            filename = fetch(lib['url'])
        if name == 'openjpeg':
            for compiler in all_compilers():
                if not os.path.exists(os.path.join(
                        build_dir, lib['dir']+compiler['inc_dir'])):
                    extract(filename, build_dir)
                    os.rename(os.path.join(build_dir, lib['dir']),
                              os.path.join(
                                  build_dir, lib['dir']+compiler['inc_dir']))
        else:
            extract(filename, build_dir)


def extract_openjpeg(compiler):
    return r"""
rem build openjpeg
setlocal
@echo on
cd %%BUILD%%
mkdir %%INCLIB%%\openjpeg-2.0
copy /Y /B openjpeg-2.0.0-win32-x86\include\openjpeg-2.0  %%INCLIB%%\openjpeg-2.0
copy /Y /B openjpeg-2.0.0-win32-x86\bin\  %%INCLIB%%
copy /Y /B openjpeg-2.0.0-win32-x86\lib\  %%INCLIB%%
endlocal
""" % compiler


def cp_tk(ver_85, ver_86):
    versions = {'ver_85': ver_85, 'ver_86': ver_86}
    return r"""
mkdir %%INCLIB%%\tcl85\include\X11
copy /Y /B %%BUILD%%\tcl%(ver_85)s\generic\*.h %%INCLIB%%\tcl85\include\
copy /Y /B %%BUILD%%\tk%(ver_85)s\generic\*.h %%INCLIB%%\tcl85\include\
copy /Y /B %%BUILD%%\tk%(ver_85)s\xlib\X11\* %%INCLIB%%\tcl85\include\X11\

mkdir %%INCLIB%%\tcl86\include\X11
copy /Y /B %%BUILD%%\tcl%(ver_86)s\generic\*.h %%INCLIB%%\tcl86\include\
copy /Y /B %%BUILD%%\tk%(ver_86)s\generic\*.h %%INCLIB%%\tcl86\include\
copy /Y /B %%BUILD%%\tk%(ver_86)s\xlib\X11\* %%INCLIB%%\tcl86\include\X11\
""" % versions


def header():
    return r"""setlocal
set MSBUILD=C:\Windows\Microsoft.NET\Framework64\v4.0.30319\MSBuild.exe
set CMAKE="cmake.exe"
set INCLIB=%~dp0\depends
set BUILD=%~dp0\build
""" + "\n".join(r'set %s=%%BUILD%%\%s' % (k.upper(), v['dir'])
                for (k, v) in libs.items() if v['dir'])


def setup_compiler(compiler):
    return r"""setlocal EnableDelayedExpansion
call "%%ProgramFiles%%\Microsoft SDKs\Windows\%(env_version)s\Bin\SetEnv.Cmd" /Release %(env_flags)s
set INCLIB=%%INCLIB%%\%(inc_dir)s
""" % compiler  # noqa: E501


def end_compiler():
    return """
endlocal
"""


def nmake_openjpeg(compiler):
    atts = {'op_ver': '2.1'}
    atts.update(compiler)
    return r"""
rem build openjpeg
setlocal
@echo on
cd /D %%OPENJPEG%%%(inc_dir)s

%%CMAKE%% -DBUILD_THIRDPARTY:BOOL=OFF -G "NMake Makefiles" .
nmake -f Makefile clean
nmake -f Makefile
copy /Y /B bin\* %%INCLIB%%
mkdir %%INCLIB%%\openjpeg-%(op_ver)s
copy /Y /B src\lib\openjp2\*.h %%INCLIB%%\openjpeg-%(op_ver)s
endlocal
""" % atts


def nmake_libs(compiler, bit):
    # undone -- pre, makes, headers, libs
    script = r"""
rem Build libjpeg
setlocal
""" + vc_setup(compiler, bit) + r"""
cd /D %%JPEG%%
nmake -f makefile.vc setup-vc6
nmake -f makefile.vc clean
nmake -f makefile.vc libjpeg.lib
copy /Y /B *.dll %%INCLIB%%
copy /Y /B *.lib %%INCLIB%%
copy /Y /B j*.h %%INCLIB%%
endlocal

rem Build zlib
setlocal
cd /D %%ZLIB%%
nmake -f win32\Makefile.msc clean
nmake -f win32\Makefile.msc zlib.lib
copy /Y /B *.dll %%INCLIB%%
copy /Y /B *.lib %%INCLIB%%
copy /Y /B zlib.lib %%INCLIB%%\z.lib
copy /Y /B zlib.h %%INCLIB%%
copy /Y /B zconf.h %%INCLIB%%
endlocal

rem Build webp
setlocal
""" + vc_setup(compiler, bit) + r"""
cd /D %%WEBP%%
rd /S /Q %%WEBP%%\output\release-static
nmake -f Makefile.vc CFG=release-static RTLIBCFG=static OBJDIR=output all
copy /Y /B output\release-static\%(webp_platform)s\lib\* %%INCLIB%%
mkdir %%INCLIB%%\webp
copy /Y /B src\webp\*.h %%INCLIB%%\\webp
endlocal

rem Build libtiff
setlocal
""" + vc_setup(compiler, bit) + r"""
rem do after building jpeg and zlib
copy %%~dp0\nmake.opt %%TIFF%%

cd /D %%TIFF%%
nmake -f makefile.vc clean
nmake -f makefile.vc lib
copy /Y /B libtiff\*.dll %%INCLIB%%
copy /Y /B libtiff\*.lib %%INCLIB%%
copy /Y /B libtiff\tiff*.h %%INCLIB%%
endlocal
"""
    return script % compiler


def msbuild_freetype(compiler, bit):
    script = r"""
rem Build freetype
setlocal
rd /S /Q %%FREETYPE%%\objs
set DefaultPlatformToolset=v100
"""
    properties = r"""/p:Configuration="Release" /p:Platform=%(platform)s"""
    if bit == 64:
        script += r'copy /Y /B ' +\
            r'"C:\Program Files (x86)\Microsoft SDKs\Windows\v7.1A\Lib\x64\*.Lib" ' +\
            r'%%FREETYPE%%\builds\windows\vc2010'
        properties += r" /p:_IsNativeEnvironment=false"
    script += r"""
%%MSBUILD%% %%FREETYPE%%\builds\windows\vc2010\freetype.sln /t:Clean;Build """+properties+r""" /m
xcopy /Y /E /Q %%FREETYPE%%\include %%INCLIB%%
"""
    freetypeReleaseDir = r"%%FREETYPE%%\objs\%(platform)s\Release"
    script += r"""
copy /Y /B """+freetypeReleaseDir+r"""\freetype.lib %%INCLIB%%\freetype.lib
copy /Y /B """+freetypeReleaseDir+r"""\freetype.dll %%INCLIB%%\..\freetype.dll
endlocal
"""
    return script % compiler  # noqa: E501


def build_lcms2(compiler):
    if compiler['env_version'] == 'v7.1':
        return build_lcms_71(compiler)
    return build_lcms_70(compiler)


def build_lcms_70(compiler):
    """Link error here on x64"""
    if compiler['platform'] == 'x64':
        return ''

    """Build LCMS on VC2008. This version is only 32bit/Win32"""
    return r"""
rem Build lcms2
setlocal
rd /S /Q %%LCMS%%\Lib
rd /S /Q %%LCMS%%\Projects\VC%(vc_version)s\Release
%%MSBUILD%% %%LCMS%%\Projects\VC%(vc_version)s\lcms2.sln /t:Clean /p:Configuration="Release" /p:Platform=Win32 /m
%%MSBUILD%% %%LCMS%%\Projects\VC%(vc_version)s\lcms2.sln /t:lcms2_static /p:Configuration="Release" /p:Platform=Win32 /p:PlatformToolset=v90 /m
xcopy /Y /E /Q %%LCMS%%\include %%INCLIB%%
copy /Y /B %%LCMS%%\Lib\MS\*.lib %%INCLIB%%
endlocal
""" % compiler  # noqa: E501


def build_lcms_71(compiler):
    return r"""
rem Build lcms2
setlocal
rd /S /Q %%LCMS%%\Lib
rd /S /Q %%LCMS%%\Projects\VC%(vc_version)s\Release
%%MSBUILD%% %%LCMS%%\Projects\VC%(vc_version)s\lcms2.sln /t:Clean /p:Configuration="Release" /p:Platform=%(platform)s /m
%%MSBUILD%% %%LCMS%%\Projects\VC%(vc_version)s\lcms2.sln /t:lcms2_static /p:Configuration="Release" /p:Platform=%(platform)s /m
xcopy /Y /E /Q %%LCMS%%\include %%INCLIB%%
copy /Y /B %%LCMS%%\Lib\MS\*.lib %%INCLIB%%
endlocal
""" % compiler  # noqa: E501


def build_ghostscript(compiler, bit):
    script = r"""
rem Build gs
setlocal
""" + vc_setup(compiler, bit) + r"""
set MSVC_VERSION=""" + {
        "2010": "90",
        "2015": "14"
    }[compiler['vc_version']] + r"""
set RCOMP="C:\Program Files (x86)\Microsoft SDKs\Windows\v7.1A\Bin\RC.Exe"
cd /D %%GHOSTSCRIPT%%
"""
    if bit == 64:
        script += r"""
set WIN64=""
"""
    script += r"""
nmake -f psi/msvc.mak
copy /Y /B bin\ C:\Python27\
endlocal
"""
    return script % compiler  # noqa: E501


def add_compiler(compiler, bit):
    script.append(setup_compiler(compiler))
    script.append(nmake_libs(compiler, bit))

    # script.append(extract_openjpeg(compiler))

    script.append(msbuild_freetype(compiler, bit))
    script.append(build_lcms2(compiler))
    # script.append(nmake_openjpeg(compiler))
    script.append(build_ghostscript(compiler, bit))
    script.append(end_compiler())


mkdirs()
extract_libs()
script = [header(),
          cp_tk(libs['tk-8.5']['version'],
          libs['tk-8.6']['version'])]


if 'PYTHON' in os.environ:
    add_compiler(compiler_from_env(), bit_from_env())
else:
    # for compiler in all_compilers():
    #     add_compiler(compiler)
    add_compiler(compilers[7.0][2010][32], 32)

with open('build_deps.cmd', 'w') as f:
    f.write("\n".join(script))
