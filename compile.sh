SRC=notes
OUT=book.pdf

echo "Compiling..."

# Compile SVGs to PDFs so pandoc can handle them.
cd $SRC/assets
for svg in *.svg; do
    rsvg-convert "$svg" -f pdf -o "${svg%.*}.pdf"
done
cd $OLDPWD

cd $SRC

# Insert pagebreaks between documents.
for f in *.md; do
    sed '$a\'$'\n''\\pagebreak'$'\n' "$f"

# Compile the notes.
# Remove newcommands (you should consolidate them elsewhere, see README.md)
# Replaces references to svg files with pdf references.
done | sed "s/^\\\newcommand.*//g" | sed "s/\(!\[.*\](.*.\)svg\()\)/\1pdf\2/g" | pandoc -s --latex-engine=xelatex --template=template.latex --mathjax -o $OLDPWD/$OUT
cd $OLDPWD

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
