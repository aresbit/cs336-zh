#!/usr/bin/env ruby
# Verify math rendering by rendering every chapter with the SAME kramdown engine
# GitHub Pages uses (kramdown + input: GFM), then reporting the three failure
# modes. This is the trustworthy oracle: kramdown's parsing differs from your
# editor's preview, so only the engine itself can confirm the math is safe.
#
# Reports per file and in total:
#   leaked $        : a span kramdown did NOT recognize as math (raw $ in output,
#                     outside code). Expect 0 — except intentionally escaped
#                     currency like \$10, which renders as a literal $ and is fine.
#   emph-in-math    : <em>/<strong> inside \(...\)/\[...\] — emphasis corruption
#                     from single-$ math. Expect 0.
#   phantom-tables  : headerless <table> produced by a stray | (real GFM tables
#                     have <thead>). Expect 0.
#
# Setup without root (no native gems needed — pure Ruby):
#   GEM_HOME=/tmp/ghp-gems gem install --no-document 'kramdown:2.4.0' \
#     kramdown-parser-gfm rexml
#   cd docs && GEM_HOME=/tmp/ghp-gems ruby /path/to/verify_math_kramdown.rb
#
# Optional first arg = glob (default 'ch*.md').

require 'kramdown'
require 'kramdown-parser-gfm'

OPTS = { input: 'GFM', hard_wrap: false, syntax_highlighter: nil }.freeze
pattern = ARGV[0] || 'ch*.md'

tot_leak = tot_emph = tot_tab = tot_inline = tot_disp = 0
files = Dir.glob(pattern).sort
abort("no files match #{pattern}") if files.empty?

files.each do |f|
  html = Kramdown::Document.new(File.read(f), **OPTS).to_html

  stripped = html.gsub(/<pre.*?<\/pre>/m, '').gsub(/<code.*?<\/code>/m, '')
  leak = stripped.scan('$').size

  inline = html.scan(/\\\(.*?\\\)/m)
  disp   = html.scan(/\\\[.*?\\\]/m)
  emph   = (inline + disp).count { |b| b =~ /<(em|strong)>/ }

  phantom = html.scan(/<table>.*?<\/table>/m).count { |t| !t.include?('<thead>') }

  tot_leak += leak; tot_emph += emph; tot_tab += phantom
  tot_inline += inline.size; tot_disp += disp.size

  flag = (leak > 0 || emph > 0 || phantom > 0) ? '  <-- CHECK' : ''
  printf("%-40s leaked$=%-3d emph-in-math=%-3d phantom-tables=%-3d  [%d inline, %d display]%s\n",
         f, leak, emph, phantom, inline.size, disp.size, flag)
end

puts '-' * 90
printf("TOTAL  leaked$=%d  emph-in-math=%d  phantom-tables=%d   recognized: %d inline + %d display\n",
       tot_leak, tot_emph, tot_tab, tot_inline, tot_disp)
puts(tot_emph.zero? && tot_tab.zero? ?
     "OK: no emphasis corruption, no phantom tables. Any leaked \$ should be intentional escaped currency." :
     "PROBLEMS above — run promote_inline_math.py and fix_math_pipes.py, then re-verify.")
exit(tot_emph.zero? && tot_tab.zero? ? 0 : 1)
