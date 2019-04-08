.. py:module:: PIL.ImageFont
.. py:currentmodule:: PIL.ImageFont

:py:mod:`ImageFont` Module
==========================

The :py:mod:`ImageFont` module defines a class with the same name. Instances of
this class store bitmap fonts, and are used with the
:py:meth:`PIL.ImageDraw.Draw.text` method.

PIL uses its own font file format to store bitmap fonts. You can use the
:command:`pilfont` utility to convert BDF and PCF font descriptors (X window
font formats) to this format.

Starting with version 1.1.4, PIL can be configured to support TrueType and
OpenType fonts (as well as other font formats supported by the FreeType
library). For earlier versions, TrueType support is only available as part of
the imToolkit package

Example
-------

.. code-block:: python

    from PIL import ImageFont, ImageDraw

    draw = ImageDraw.Draw(image)

    # use a bitmap font
    font = ImageFont.load("arial.pil")

    draw.text((10, 10), "hello", font=font)

    # use a truetype font
    font = ImageFont.truetype("arial.ttf", 15)

    draw.text((10, 25), "world", font=font)

Functions
---------

.. autofunction:: PIL.ImageFont.load
.. autofunction:: PIL.ImageFont.load_path
.. autofunction:: PIL.ImageFont.truetype
.. autofunction:: PIL.ImageFont.load_default

Methods
-------

.. py:method:: PIL.ImageFont.ImageFont.getsize(text, direction=None, features=[], language=None)

    Returns width and height (in pixels) of given text if rendered in font with
    provided direction, features, and language.
    
    :param text: Text to measure.
    
    :param direction: Direction of the text. It can be 'rtl' (right to
                      left), 'ltr' (left to right) or 'ttb' (top to bottom).
                      Requires libraqm.

                      .. versionadded:: 4.2.0

    :param features: A list of OpenType font features to be used during text
                     layout. This is usually used to turn on optional
                     font features that are not enabled by default,
                     for example 'dlig' or 'ss01', but can be also
                     used to turn off default font features for
                     example '-liga' to disable ligatures or '-kern'
                     to disable kerning.  To get all supported
                     features, see
                     https://docs.microsoft.com/en-us/typography/opentype/spec/featurelist
                     Requires libraqm.

                     .. versionadded:: 4.2.0

    :param language: Language of the text. Different languages may use 
                     different glyph shapes or ligatures. This parameter tells
                     the font which language the text is in, and to apply the
                     correct substitutions as appropriate, if available.
                     It should be a `BCP47 language code
                     <https://www.w3.org/International/articles/language-tags/>`
                     Requires libraqm.

                     .. versionadded:: 6.0.0

    :return: (width, height)

.. py:method:: PIL.ImageFont.ImageFont.getmask(text, mode='', direction=None, features=[], language=None)

    Create a bitmap for the text.

    If the font uses antialiasing, the bitmap should have mode “L” and use a
    maximum value of 255. Otherwise, it should have mode “1”.

    :param text: Text to render.
    :param mode: Used by some graphics drivers to indicate what mode the
                 driver prefers; if empty, the renderer may return either
                 mode. Note that the mode is always a string, to simplify
                 C-level implementations.

                 .. versionadded:: 1.1.5

    :param direction: Direction of the text. It can be 'rtl' (right to
                      left), 'ltr' (left to right) or 'ttb' (top to bottom).
                      Requires libraqm.

                      .. versionadded:: 4.2.0

    :param features: A list of OpenType font features to be used during text
                     layout. This is usually used to turn on optional
                     font features that are not enabled by default,
                     for example 'dlig' or 'ss01', but can be also
                     used to turn off default font features for
                     example '-liga' to disable ligatures or '-kern'
                     to disable kerning.  To get all supported
                     features, see
                     https://docs.microsoft.com/en-us/typography/opentype/spec/featurelist
                     Requires libraqm.

                     .. versionadded:: 4.2.0

    :param language: Language of the text. Different languages may use 
                     different glyph shapes or ligatures. This parameter tells
                     the font which language the text is in, and to apply the
                     correct substitutions as appropriate, if available.
                     It should be a `BCP47 language code
                     <https://www.w3.org/International/articles/language-tags/>`
                     Requires libraqm.

                     .. versionadded:: 6.0.0

    :return: An internal PIL storage memory instance as defined by the
             :py:mod:`PIL.Image.core` interface module.
