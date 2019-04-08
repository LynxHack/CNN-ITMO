#!/bin/sh

mkdir /var/cache/pacman/pkg
pacman -S --noconfirm mingw32/mingw-w64-i686-python3-pip \
	   mingw32/mingw-w64-i686-python3-setuptools \
	   mingw32/mingw-w64-i686-python2-pip \
	   mingw32/mingw-w64-i686-python2-setuptools \
	   mingw-w64-i686-libjpeg-turbo

C:/msys64/mingw32/bin/python3 -m pip install --upgrade pip

/mingw32/bin/pip install pytest pytest-cov olefile
/mingw32/bin/pip3 install pytest pytest-cov olefile
