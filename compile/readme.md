
# Compiling

To compile, just run `./compile.sh <pdf|html>`.

To compile the HTML version, you must have the following installed:

    sudo pip install pyyaml py-gfm markdown

in a Python 3 environment (you can use `virtualenv`).

To compile the PDF, you need the following prerequisites:

OSX:

    brew install pandoc librsvg
    # Also install: [MacTex](https://tug.org/mactex/)

Linux (using `apt`):

    sudo apt-get install librsvg2-bin lmodern

    # The texlive and texlive-xetex apt packages for Ubuntu 14.04 are old af
    # So manually install the latest:
    wget http://mirror.ctan.org/systems/texlive/tlnet/install-tl-unx.tar.gz
    tar -xzvf install-tl-unx.tar.gz
    cd install-tl-*

    # Enter "i" when prompted
    # fyi this is a massive download
    sudo ./install-tl

    # Then add:
    # /usr/local/texlive/2015/bin/x86_64-linux/
    # or its equivalent to your PATH.

    # Then you should be able to do
    # tlmgr update --self
    # tlmgr update --all
    # to stay up-to-date.

    # Check the versions with
    tex --version
    latex --version
    xelatex --version
    pdflatex --version

    # Refer to:
    # - <https://www.tug.org/texlive/acquire-netinstall.html>
    # - <https://www.tug.org/texlive/quickinstall.html>

    # Also worth getting the latest pandoc from:
    # https://github.com/jgm/pandoc/releases
    # then install with gdebi:
    # sudo gdebi pandoc-*.deb


