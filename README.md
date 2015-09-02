This is a collection of my machine learning and artificial intelligence notes. It's very much a work-in-progress and large portions are very disorganized. I have tried to be diligent about references but a lot of them, especially earlier ones, may have fallen through the cracks...if I forgot to cite someone, let me know or submit a PR.

The source files for the graphics are in the `graphics` folder, which includes an iPython notebook for the graphics generated with `matplotlib`/`seaborn` and Illustrator files for the others (in the `illustrations` directory).

I have not included the fonts (some of them are copyrighted or something) - in a later update I will use open source fonts so they aren't an issue. For now, just refer to the pre-compiled PDF, `notes.pdf`.

## Compiling

Compile prereqs:

OSX:

    brew install pandoc librsvg
    # Also: [MacTex](https://tug.org/mactex/)

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

- don't have empty line breaks in your mathjax blocks, latex will fail on them
- if you have any `newcommand`s, include them in the individual markdown files that need them. The script will automatically remove these (since redundant ones mess up latex), so also set up a yaml header for pandoc to use in one of your markdown files, for example:

```
---
title: Artificial Intelligence Notes
author: Francis Tseng
header-includes:
    - \newcommand{\argmax}{\operatorname*{argmax}}
    - \newcommand{\argmin}{\operatorname*{argmin}}
toc: yes
abstract: "This is my abstract"
---
```


Refer to:

- <https://tex.stackexchange.com/questions/139139/adding-headers-and-footers-using-pandoc>
- <http://pandoc.org/demo/example9/templates.html>
