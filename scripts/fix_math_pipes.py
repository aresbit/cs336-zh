#!/usr/bin/env python3
r"""Replace bare `|` inside $$...$$ math with `\vert` so kramdown (GFM) does not
mis-parse the line as a table.

Why: kramdown's GFM parser turns ANY line containing a `|` into a table row —
even plain prose. When a `|` sits inside an inline math span, the line is split
into <td> cells, tearing the $$...$$ pair apart. `\vert` renders identically to
`|` in MathJax but contains no pipe character, so the table parser leaves the
line alone. Existing `\|` (LaTeX norm) is already escaped and safe, so it is left
untouched.

Scope: replaces bare `|` (not preceded by a backslash) everywhere inside math,
including multi-line $$...$$ blocks. This is uniform and harmless — `\vert`
renders the same as `|` in every context (abs value, conditional, cardinality),
and `\left|` correctly becomes `\left\vert`. Switch specific spots to `\mid` /
`\lvert\rvert` by hand where relation/delimiter spacing matters.

Usage (from the docs/ directory):
  python fix_math_pipes.py            # dry run
  python fix_math_pipes.py --apply
  python fix_math_pipes.py --apply --glob 'ch*.md'
"""
import sys, glob


def convert(text):
    out = []
    in_fence = False
    in_math = False
    at_line_start = True
    i, n = 0, len(text)
    while i < n:
        if at_line_start and not in_math and text.startswith('```', i):
            j = text.find('\n', i)
            j = n if j == -1 else j
            out.append(text[i:j])
            in_fence = not in_fence
            i = j
            at_line_start = False
            continue
        if in_fence:
            c = text[i]
            out.append(c)
            at_line_start = (c == '\n')
            i += 1
            continue
        if text.startswith('$$', i):
            out.append('$$')
            in_math = not in_math
            i += 2
            at_line_start = False
            continue
        c = text[i]
        if in_math and c == '|' and (not out or out[-1] != '\\'):
            out.append('\\vert ')
        else:
            out.append(c)
        at_line_start = (c == '\n')
        i += 1
    return ''.join(out)


def main():
    apply = '--apply' in sys.argv
    pattern = 'ch*.md'
    if '--glob' in sys.argv:
        pattern = sys.argv[sys.argv.index('--glob') + 1]
    for f in sorted(glob.glob(pattern)):
        src = open(f, encoding='utf-8').read()
        dst = convert(src)
        if dst != src:
            print(f"{f}: {src.count('|') - dst.count('|')} bare pipe(s) -> \\vert")
            if apply:
                open(f, 'w', encoding='utf-8').write(dst)
        else:
            print(f"{f}: same")
    print("WRITTEN" if apply else "DRY RUN (use --apply to write)")


if __name__ == '__main__':
    main()
