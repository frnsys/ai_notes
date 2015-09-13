DIR=$PWD
SRC=$DIR/notes
OUT=$DIR/notes.pdf
CMP=$DIR/compile
WRK=/tmp/ml_ai_notes
TEX=$WRK/notes.tex


function compile_pdf {
    # Compile SVGs to PDFs so pandoc can handle them.
    cd $SRC/assets
    for svg in *.svg; do
        rsvg-convert "$svg" -f pdf -o "${svg%.*}.pdf"
    done
    cd $SRC

    # Create working directory
    mkdir -p $WRK

    # Insert pagebreaks between documents.
    for f in *.md; do
        sed '$a\'$'\n''\\pagebreak'$'\n' "$f"

    done > $WRK/ml_ai_notes.md

    # Copy things over to working directory (bleh)
    cp $CMP/template.latex $WRK/
    cp $CMP/fonts/{*.otf,*.ttf} $WRK/
    cp $CMP/postprocess.py $WRK/
    ln -sfn $SRC/assets/ $WRK/assets
    cd $WRK

    # Process markdown file
    python $WRK/postprocess.py $WRK/ml_ai_notes.md $WRK/tmp.md

    # Compile to .tex intermediary
    pandoc -s tmp.md --latex-engine=xelatex --template=template.latex --mathjax --highlight-style=pygments --chapters -o $TEX
    python $WRK/postprocess.py $TEX $WRK/tmp.tex

    # Compile to PDF
    xelatex $WRK/tmp.tex
    mv tmp.pdf $OUT

    # sed command breakdown:
    # First matching group:     \(!\[.*\](.*.\)
    #   matches ![.*](.*
    # Replacement target:       svg
    # Second matching group:    \()\)
    #   just matches the last parenthesis
    # Replacement:
    #   \1  => preserves first matching group
    #   pdf => replaces svg with pdf
    #   \2  => preserves second matching group
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
fi


echo "Compiled."
