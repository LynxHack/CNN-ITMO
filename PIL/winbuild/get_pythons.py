from fetch import fetch
import os

if __name__ == '__main__':
    for version in ['2.7.15', '3.4.4']:
        for platform in ['', '.amd64']:
            for extension in ['', '.asc']:
                fetch('https://www.python.org/ftp/python/%s/python-%s%s.msi%s'
                      % (version, version, platform, extension))

    # find pip, if it's not in the path!
    os.system('pip install virtualenv')
