
# Graphics

These are the source files for the graphics. It includes an iPython notebook for the graphics generated with `matplotlib`/`seaborn` and Illustrator files for the others (in the `illustrations` directory).

When exporting Illustrator files as SVG, you should set "Type: Convert to Outline" to preserve the fonts.

To use LaTeX equations with Adobe Illustrator, use LaTeXiT!, and in the General Tab of its Preferences, select "PDF with outlined fonts". Then you can type in LaTeX equations, hit "LaTeX it!", and drag-and-drop the result into Illustrator. [tip from here](https://www.quora.com/How-do-I-import-LaTeX-equations-and-symbols-into-Adobe-Illustrator)

This is the preamble for LaTeXiT! (set in Preferences > Templates):

    \documentclass[10pt]{article}
    \usepackage[usenames]{color} %used for font color
    \usepackage{amssymb} %maths
    \usepackage{amsmath} %maths
    \usepackage{cmbright}
    \usepackage[utf8]{inputenc} %useful to type directly diacritic characters

