#!/usr/bin/env python3
"""Promote single-$ inline math -> $$...$$ so kramdown (GFM) recognizes it as math
and stops parsing underscores/asterisks inside as Markdown emphasis.

Why: kramdown's only math delimiter is `$$` (inline AND display). A single `$`
span is treated as ordinary text, so `x_i ... y_j` underscores flank into <em>,
destroying subscripts before MathJax ever runs. Converting to `$$` makes kramdown
treat the content as opaque math and emit clean \\(...\\) / \\[...\\].

Safety:
  - Fenced ``` code blocks are left untouched (math/pipe rules don't apply there).
  - Existing $$...$$ (inline or multi-line display) is preserved verbatim.
  - Asserts even single-$ parity per non-code segment, so a stray literal $
    (e.g. unescaped currency) is caught instead of silently mispairing the file.

Usage (from the docs/ directory):
  python promote_inline_math.py            # dry run, lists changes
  python promote_inline_math.py --apply    # write changes
  python promote_inline_math.py --apply --glob 'ch*.md'
"""
import sys, re, glob

FENCE = re.compile(r'(^```.*?^```)', re.DOTALL | re.MULTILINE)
DD = '\x00'   # placeholder for $$  while we pair single $
ESC = '\x01'  # placeholder for \$  (escaped literal dollar — must NOT be math)


def convert_plain(seg):
    # Protect escaped currency `\$` first so it is never treated as a delimiter,
    # then protect existing $$, then pair the remaining single $ into $$.
    protected = seg.replace('\\$', ESC).replace('$$', DD)
    singles = protected.count('$')
    assert singles % 2 == 0, (
        f"odd single-$ ({singles}) — likely an unescaped literal $ "
        f"(escape it as \\$). Segment starts: {seg[:80]!r}")
    protected = re.sub(r'\$([^$\n]+?)\$', r'$$\1$$', protected)
    return protected.replace(DD, '$$').replace(ESC, '\\$')


def convert_text(text):
    out = []
    for i, part in enumerate(FENCE.split(text)):
        out.append(part if i % 2 == 1 else convert_plain(part))
    return ''.join(out)


def main():
    apply = '--apply' in sys.argv
    pattern = 'ch*.md'
    if '--glob' in sys.argv:
        pattern = sys.argv[sys.argv.index('--glob') + 1]
    for f in sorted(glob.glob(pattern)):
        src = open(f, encoding='utf-8').read()
        try:
            dst = convert_text(src)
        except AssertionError as e:
            print(f"SKIP {f}: {e}")
            continue
        if dst != src:
            print(f"{f}: CHANGED")
            if apply:
                open(f, 'w', encoding='utf-8').write(dst)
        else:
            print(f"{f}: same")
    print("WRITTEN" if apply else "DRY RUN (use --apply to write)")


if __name__ == '__main__':
    main()
