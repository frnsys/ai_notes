"""
This is hacky af but markdown/pandoc/latex is not the most friendly to customization

What this does is allow keyword arguments to be specified as part of markdown figures:

    ![My image](path/to/image.png textwidth=0.5)

This is used so the textwidth for a figure can manually be set for use in the latex document
which allows figure wrapping to be controlled from the markdown.
"""

import re
import sys
import yaml

KWARG_RE = re.compile(r'([a-z]+)=([^\s}]+)')
CLEAN_RE = re.compile(r'\s[^}]+')
CMD_RES = [re.compile(r'\\DeclareMathOperator.+'),
            re.compile(r'\\def.+')]
FRONTMATTER_RE = re.compile(r'^---\n(.*?)\n---', re.DOTALL)


def process_md(mdfile):
    # Pull out all lines
    with open(mdfile, 'r') as f:
        body = f.read()

    frontmatter = FRONTMATTER_RE.match(body)
    body = FRONTMATTER_RE.sub('', body)
    if frontmatter is not None:
        frontmatter = yaml.load(frontmatter.group(1))

        # Identify all new commands
        new_cmds = set()
        lines = body.split('\n')
        cmd_idx = []
        for i, line in enumerate(lines):
            for cmd_re in CMD_RES:
                match = cmd_re.match(line)
                if match is not None:
                    new_cmds.add(match.group(0))
                    lines[i] = ''
                    cmd_idx.append(i)

        # Clean up math delimiters for the new commands
        # Can't just use regex to identify math delimiters with just whitespace
        # b/c it's possible that there are two adjacent equations without any
        # text in between them, which is a false positive
        to_remove_idx = set()
        for i in cmd_idx:
            j = i - 1
            while True:
                if '$$' == lines[j].strip():
                    to_remove_idx.add(j)
                    break
                j -= 1

            j = i + 1
            while True:
                if '$$' == lines[j].strip():
                    to_remove_idx.add(j)
                    break
                j += 1

        for i in to_remove_idx:
            lines[i] = ''

        # Append new commands to frontmatter, then reattach
        frontmatter['header-includes'] += list(new_cmds)
        body = '\n'.join(['---', yaml.dump(frontmatter), '---'] + lines)

    return body


def process_tex(texfile):
    # Pull out all lines
    with open(texfile, 'r') as f:
        lines = [l.strip() for l in f.readlines()]

    # Customize code blocks
    for i, line in enumerate(lines):
        if r'\newenvironment{Shaded}' in lines[i]:
            # Background color
            shaded = [
                r'\usepackage{framed}',
                r'\definecolor{shadecolor}{RGB}{234,234,242}',
                r'\newenvironment{Shaded}{\begin{shaded*}}{\end{shaded*}}'
            ]
            lines[i] = '\n'.join(shaded)
            break

    # Search for graphics lines
    gfx_idx = [i for i, l in enumerate(lines) if 'includegraphics' in l]

    # Extract any keyword arguments
    for i in gfx_idx:
        kws = {k: v for k, v in KWARG_RE.findall(lines[i])}
        textwidth = kws.get('textwidth', 1.)

        # Remove the kwargs from the graphics command
        # Replace `svg` extensions with `pdf`
        lines[i] = CLEAN_RE.sub('', lines[i]).replace('.svg', '.pdf')

        # If textwidth is full (1.), just leave the figure as is
        if float(textwidth) == 1.:
            continue

        # Search for the opening and closing `figure` commands
        # for this graphic and replace as needed
        j = i - 1
        while True:
            if '\\begin{figure}' in lines[j]:
                # `R` to float images to the right
                lines[j] = '\\begin{{wrapfigure}}{{R}}{{{}\\textwidth}}'.format(textwidth)
                break
            j -= 1

        j = i + 1
        while True:
            if '\\end{figure}' in lines[j]:
                # Add a `\leavevmode` to reset mode manually, sometimes it
                # doesn't work automatically
                lines[j] = '\\end{wrapfigure}\n\\leavevmode'
                break
            j += 1

    return '\n'.join(lines)


if __name__ == '__main__':
    infile = sys.argv[1]
    outfile = sys.argv[2]
    ext = infile.split('.')[-1]

    if ext == 'tex':
        body = process_tex(infile)
    elif ext == 'md':
        body = process_md(infile)
    else:
        print('No handler for extension `{}`'.format(ext))
        sys.exit(1)

    with open(outfile, 'w') as f:
        f.write(body)