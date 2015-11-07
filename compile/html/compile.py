"""
Clean up extra latex and markdown stuff, e.g. the keyword arguments to markdown figures.
"""

import os
import re
import yaml
import shutil
from glob import glob
from md import compile_markdown
from collections import defaultdict

OUT_DIR = 'html_build'
PART_TITLE_RE = re.compile(r'\\part{([^}]+)}')
CHAPTER_TITLE_RE = re.compile(r'\s([^.]+)')
FRONTMATTER_RE = re.compile(r'^---\n(.*?)\n---', re.DOTALL)
LATEX_RE = re.compile(r'\\.+}$', re.MULTILINE)
IMAGE_RE = re.compile(r'(^!\[[^]]+\]\([^\s]+)(\s[^)]+)(\))')
ASSET_RE = re.compile(r'assets\/')


HTML_TEMPLATE = '''
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge,chrome=1">
    <meta name="viewport" content="width=device-width,initial-scale=1">

    <title>{title}</title>

    <link rel="stylesheet" type="text/css" href="{style}">
    <link rel="stylesheet" href="http://cdnjs.cloudflare.com/ajax/libs/highlight.js/8.8.0/styles/color-brewer.min.css">
</head>

<body>

    {body}
    {scripts}

</body>
</html>
'''

SCRIPTS = '''
    <script src="http://ajax.googleapis.com/ajax/libs/jquery/1.11.1/jquery.min.js"></script>
    <script src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>
    <script src="http://cdnjs.cloudflare.com/ajax/libs/highlight.js/8.8.0/highlight.min.js"></script>
    <script>
        $(document).ready(function() {
            $('pre').each(function(i, e) {hljs.highlightBlock(e)});
            MathJax.Hub.Config({
                tex2jax: {
                    inlineMath: [["$","$"]],
                    displayMath: [['$$','$$']],
                    processEscapes: true
                }
            });
            MathJax.Hub.Startup.onload();
        });
    </script>
'''

def to_html(md, title, outfile):
    path_to_top = '/'.join(['..' for i in range(len(outfile.split('/'))-1)])
    if path_to_top:
        path_to_top += '/'
    html = compile_markdown(md)
    html = HTML_TEMPLATE.format(title=title,
                                body=html,
                                scripts=SCRIPTS,
                                style='{}{}'.format(path_to_top, 'style.css'))

    out = os.path.join(OUT_DIR, outfile)
    with open(out, 'w') as f:
        f.write(html)


def process_chapter(chapter, part_slug):
    chapter_title = CHAPTER_TITLE_RE.search(chapter).group(1)
    chapter_slug = chapter_title.lower().replace(' ', '_')

    with open(chapter, 'r') as f:
        chapter_text = f.read()

        # Remove any latex `\part` commands
        chapter_text = PART_TITLE_RE.sub('', chapter_text)

        # Remove keyword arguments for images
        chapter_text = IMAGE_RE.sub(r'\g<1>\g<3>', chapter_text)

        # Update asset paths
        chapter_text = ASSET_RE.sub('../assets/', chapter_text)

        outfile = '{}/{}.html'.format(part_slug, chapter_slug)
        to_html(chapter_text, chapter_title, outfile)
        return chapter_title, outfile


if __name__ == '__main__':
    if os.path.exists(OUT_DIR):
        shutil.rmtree(OUT_DIR)

    # Sort notes into their parts
    parts = defaultdict(list)
    curr_part = 0
    for file in glob('notes/*.md'):
        note = os.path.basename(file)
        part = int(note.split('.')[0])
        parts[part].append(file)

    # Sort parts, keep file lists
    parts = [v for k, v in sorted(parts.items())]
    part_files = []

    # Skip the cover
    for part in parts[1:]:
        # Sort the part's chapters
        part = sorted(part)
        chapter_paths = []

        # The first "chapter" is the part divider file
        with open(part[0], 'r') as f:
            chapter_text = f.read()

            # Extract part title
            part_title = PART_TITLE_RE.search(chapter_text)
            if part_title is None:
                raise Exception('Could not find part title for part {}'.format(part[0]))

            part_title = part_title.group(1)
            part_slug = part_title.lower().replace(' ', '_')
            os.makedirs(os.path.join(OUT_DIR, part_slug))

        for chapter in part[1:]:
            chap_title, chap_path = process_chapter(chapter, part_slug)
            chapter_paths.append((chap_title, chap_path))

        part_files.append((part_title, chapter_paths))

    # Sorry folks, we are building the table of contents by hand!
    toc = []
    for part_title, chapter_paths in part_files:
        part_heading = '- {}'.format(part_title)
        chapter_links = ['    - [{}]({})'.format(ch_t, ch_p) for ch_t, ch_p in chapter_paths]
        toc.append('\n'.join([part_heading]+chapter_links))

    toc = '\n'.join(toc)

    # The first part becomes the index page
    intro = parts[0][0]
    with open(intro, 'r') as f:
        intro_text = f.read()

        # Convert frontmatter into markdown
        frontmatter = FRONTMATTER_RE.match(intro_text)
        frontmatter = yaml.load(frontmatter.group(1))
        author = frontmatter['author']
        subtitle = frontmatter['subtitle']
        title = frontmatter['title']

        intro_text = FRONTMATTER_RE.sub('', intro_text)
        intro_text = LATEX_RE.sub('', intro_text)
        intro_text = '# {title}\n## {author}\n#### {subtitle}\n\n{body}\n\n{toc}'.format(
            title=title,
            author=author,
            subtitle=subtitle,
            body=intro_text,
            toc=toc
        )
    to_html(intro_text, title, 'index.html')

    # Copy over assets
    shutil.copytree('notes/assets', os.path.join(OUT_DIR, 'assets'))
    shutil.copy('compile/html/style.css', os.path.join(OUT_DIR, 'style.css'))