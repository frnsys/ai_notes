__To see the notes, look at `notes.pdf`__

This is a collection of my machine learning and artificial intelligence notes. It's very much a work-in-progress and large portions are very disorganized. I have tried to be diligent about references but a lot of them, especially earlier ones, may have fallen through the cracks...if I forgot to cite someone, let me know or submit a PR.


## Graphics

The source files for the graphics are in the `graphics` folder, which includes an iPython notebook for the graphics generated with `matplotlib`/`seaborn` and Illustrator files for the others (in the `illustrations` directory).

When exporting Illustrator files as SVG, you should set "Type: Convert to Outline" to preserve the fonts.

## Styling

- Main font family: Lato
- Text font family: Source Sans Pro
- Figure sans-serif font family: Calibre
- Math: cmbright

To use LaTeX equations with Adobe Illustrator, use LaTeXiT!, and in the General Tab of its Preferences, select "PDF with outlined fonts". Then you can type in LaTeX equations, hit "LaTeX it!", and drag-and-drop the result into Illustrator. [tip from here](https://www.quora.com/How-do-I-import-LaTeX-equations-and-symbols-into-Adobe-Illustrator)

This is the preamble for LaTeXiT! (set in Preferences > Templates):

    \documentclass[10pt]{article}
    \usepackage[usenames]{color} %used for font color
    \usepackage{amssymb} %maths
    \usepackage{amsmath} %maths
    \usepackage{cmbright}
    \usepackage[utf8]{inputenc} %useful to type directly diacritic characters


## Conventions

- New terms are introduced in __bold__


## Compiling

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

## Notes

- Don't have empty line breaks in your MathJax blocks, LaTeX will fail on them.
- If you need any custom math commands, use either `\DeclareMathOperator` or `\providecommand`. The postprocessing script (`compile/postprocess.py`) will identify these, hoist them to the YAML frontmatter, and properly clean up redundant definitions.

Refer to:

- <https://tex.stackexchange.com/questions/139139/adding-headers-and-footers-using-pandoc>
- <http://pandoc.org/demo/example9/templates.html>
