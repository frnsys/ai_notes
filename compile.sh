DIR=$PWD
SRC=$DIR/notes
OUT=$DIR/notes.pdf
CMP=$DIR/compile
WRK=/tmp/ml_ai_notes

echo "Compiling..."

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

# Compile the notes.
# Remove newcommands (you should consolidate them elsewhere, see README.md)
# Replaces references to svg files with pdf references.
done | sed "s/^\\\newcommand.*//g" | sed "s/\(!\[.*\](.*.\)svg\()\)/\1pdf\2/g" > $WRK/ml_ai_notes.md

# Copy things over to working directory (bleh)
cp $CMP/template.latex $WRK/
cp $CMP/fonts/{*.otf,*.ttf} $WRK/

#rm $WRK/assets
ln -sfn $SRC/assets/ $WRK/assets

cd $WRK
pandoc -s ml_ai_notes.md --latex-engine=xelatex --template=template.latex --mathjax -o $OUT

echo "Compiled."

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
