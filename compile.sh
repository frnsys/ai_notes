DIR=$PWD
SRC=$DIR/notes
OUT=$DIR/notes.pdf
CMP=$DIR/compile
WRK=/tmp/ai_notes
TEX=$WRK/notes.tex


function compile_pdf {
    # Compile SVGs to PDFs so pandoc can handle them.
    cd $SRC/assets
    for svg in *.svg; do
        rsvg-convert "$svg" -f pdf -o "${svg%.*}.pdf"
    done
    cd $SRC

    # Create working directory
    rm -rf $WRK
    mkdir -p $WRK

    # Insert pagebreaks between documents.
    for f in *.md; do
        sed '$a\'$'\n''\\pagebreak'$'\n' "$f"

    done > $WRK/ai_notes.md

    # Copy things over to working directory (bleh)
    cp $CMP/template.latex $WRK/
    cp $CMP/fonts/{*.otf,*.ttf} $WRK/
    cp $CMP/postprocess.py $WRK/
    ln -sfn $SRC/assets/ $WRK/assets
    cd $WRK

    # Process markdown file
    python $WRK/postprocess.py $WRK/ai_notes.md $WRK/tmp.md

    # Compile to .tex intermediary
    pandoc -s tmp.md --latex-engine=xelatex --template=template.latex --mathjax --highlight-style=pygments --chapters -o $TEX
    python $WRK/postprocess.py $TEX $WRK/tmp.tex

    # Compile to PDF
    # Run twice so table of contents works properly, see:
    # <http://stackoverflow.com/questions/3863630/latex-tableofcontents-command-always-shows-blank-contents-on-first-build>
    xelatex $WRK/tmp.tex
    xelatex $WRK/tmp.tex
    mv tmp.pdf $OUT
}

function compile_html {
    python compile/html/compile.py
}


if [ -z $1 ]; then
    echo -e "$(tput setaf 3)Specify 'pdf' or 'html'$(tput sgr0)"
    exit
fi

echo "Compiling..."

if [ $1 == 'pdf' ]; then
    compile_pdf

elif [ $1 == 'html' ]; then
    compile_html

else
    echo "Unrecognized format, please specify 'pdf' or 'html'"
    exit
fi

echo "Compiled."
