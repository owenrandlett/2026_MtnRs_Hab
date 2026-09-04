# preprocess.awk -- rewrite the manuscript .tex into pandoc-ready LaTeX.
#
#   gawk -v meta=meta.yaml -f preprocess.awk 2026_MtnRHab.tex > body.tex
#
# Figure and table numbers are read off the \label that FOLLOWS each \caption
# or \figsupp, and in-text \autoref/\autorefp are rewritten from the same
# labels.  Caption numbers and cross-references therefore cannot drift apart,
# and adding or reordering a figure needs no change here.

function balanced(s, start,   i, d, c, out) {
    # s must have "{" at position `start`; return the matching brace group.
    d = 0
    for (i = start; i <= length(s); i++) {
        c = substr(s, i, 1)
        if (c == "{")      { d++; if (d == 1) continue }
        else if (c == "}") { d--; if (d == 0) return out }
        out = out c
    }
    return out
}

function tomd(s) {
    # Just enough LaTeX -> markdown for the YAML metadata fields.
    s = gensub(/\\textsubscript[{]([^}]*)[}]/,  "~\\1~",   "g", s)
    s = gensub(/\\textsuperscript[{]([^}]*)[}]/, "^\\1^",  "g", s)
    s = gensub(/\\emph[{]([^}]*)[}]/,           "*\\1*",   "g", s)
    s = gensub(/\\textbf[{]([^}]*)[}]/,         "**\\1**", "g", s)
    gsub(/\n/, " ", s)
    gsub(/[[:space:]]+/, " ", s)
    gsub(/^ +| +$/, "", s)
    gsub(/'/, "''", s)                 # YAML single-quote escaping
    return s
}

function inject_figsupp(startline, num,   i, s, pos, c, depth, seenopt) {
    # \figsupp[short]{caption}{graphic}: insert the number at the front of the
    # caption argument, so the shim can render it as image-then-caption and
    # match the layout of the main figures.
    i = startline
    s = L[i]
    pos = index(s, "\\figsupp") + 8
    depth = 0
    seenopt = 0
    while (i <= n) {
        while (pos <= length(s)) {
            c = substr(s, pos, 1)
            if (seenopt) {
                if (c == "{") break
            } else if (c == "[") depth++
            else if (c == "]") { depth--; if (depth == 0) seenopt = 1 }
            else if (c == "{" && depth == 0) break   # no optional argument
            pos++
        }
        if (pos <= length(s)) {
            L[i] = substr(s, 1, pos) "\\textbf{" num ".} " substr(s, pos + 1)
            return
        }
        i++
        s = L[i]
        pos = 1
    }
}

function expand_mc(r,   m, n, i, pre, rest, spec, txt, tail, rep, guard) {
    # \multicolumn{n}{spec}{TEXT} -> TEXT repeated across the n cells it spans,
    # so the row can be merged cell-by-cell with the rows above and below it.
    guard = 0
    while (match(r, /\\multicolumn[{]([0-9]+)[}]/, m) && guard++ < 50) {
        n    = m[1] + 0
        pre  = substr(r, 1, RSTART - 1)
        rest = substr(r, RSTART + RLENGTH)      # "{spec}{TEXT}..."
        spec = balanced(rest, 1)
        rest = substr(rest, length(spec) + 3)   # "{TEXT}..."
        txt  = balanced(rest, 1)
        tail = substr(rest, length(txt) + 3)
        rep  = txt
        for (i = 2; i <= n; i++) rep = rep " & " txt
        r = pre rep tail
    }
    return r
}

function flatten_header(s,   nr, rows, i, j, nc, cc, v, r, maxc, out, merged) {
    # Pandoc's LaTeX reader keeps only a single header row -- given two it
    # silently drops the header and renders every row as a body row.  Merge
    # the rows column-wise into one so the header survives.
    delete merged
    maxc = 0
    nr = split(s, rows, /\\\\/)
    for (i = 1; i <= nr; i++) {
        r = expand_mc(rows[i])
        if (r ~ /^[[:space:]]*$/) continue
        nc = split(r, cc, /&/)
        if (nc > maxc) maxc = nc
        for (j = 1; j <= nc; j++) {
            v = cc[j]
            gsub(/[[:space:]]+/, " ", v)
            gsub(/^ +| +$/, "", v)
            if (v == "") continue
            merged[j] = (j in merged) ? merged[j] " " v : v
        }
    }
    for (j = 1; j <= maxc; j++)
        out = (j == 1 ? merged[j] : out " & " merged[j])
    return out " \\\\"
}

function fixrefs(s) {
    # \autorefp{fig:3}{B} -> **Figure 3B**   \autoref{fig:S2} -> **Figure S2**
    s = gensub(/\\autorefp[{]fig:([^}]*)[}][{]([^}]*)[}]/, "\\\\textbf{Figure \\1\\2}", "g", s)
    s = gensub(/\\autoref[{]fig:([^}]*)[}]/,               "\\\\textbf{Figure \\1}",    "g", s)
    s = gensub(/\\autoref[{]supptab:[^}]*[}]/,             "\\\\textbf{Supplementary Table 1}", "g", s)
    s = gensub(/\\ref[{]supptab:[^}]*[}]/,                 "1", "g", s)
    return s
}

{
    L[++n] = $0
    if ($0 ~ /\\begin[{]document[}]/) bodystart = n
    # Preamble, minus comment lines, is scanned for title/authors/affiliations.
    else if (!bodystart && $0 !~ /^[[:space:]]*%/) pre = pre $0 "\n"
}

END {
    # ---- 1. number each \caption / \figsupp from the \label that follows ----
    for (i = 1; i <= n; i++) {
        if (L[i] ~ /\\caption[{]/)        { pend = i; pendtype = "caption" }
        else if (L[i] ~ /\\figsupp[[]/)   { pend = i; pendtype = "figsupp" }
        else if (pend && match(L[i], /\\label[{](fig|supptab):([^}]*)[}]/, m)) {
            if (m[1] == "fig") num = "Figure " m[2]
            else               num = "Supplementary Table " ++ntab
            if (pendtype == "caption")
                sub(/\\caption[{]/, "\\caption{\\textbf{" num ".} ", L[pend])
            else
                inject_figsupp(pend, num)
            pend = 0
        }
    }

    # ---- 2. front matter -> metadata file ----
    if (match(pre, /\\title[{]/))
        title = balanced(pre, RSTART + RLENGTH - 1)

    p = pre
    while (match(p, /\\author[[]([^]]*)[]][[:space:]]*[{]/)) {
        ids = gensub(/^\\author[[]([^]]*)[]].*$/, "\\1", 1, substr(p, RSTART, RLENGTH))
        gsub(/[[:space:]]/, "", ids)
        bp = RSTART + RLENGTH - 1
        name = balanced(p, bp)
        auth[++na] = tomd(name) "^" ids "^"
        p = substr(p, bp + length(name) + 2)
    }

    p = pre
    while (match(p, /\\affil[[]([^]]*)[]][[:space:]]*[{]/)) {
        ids = gensub(/^\\affil[[]([^]]*)[]].*$/, "\\1", 1, substr(p, RSTART, RLENGTH))
        gsub(/[[:space:]]/, "", ids)
        bp = RSTART + RLENGTH - 1
        body = balanced(p, bp)
        gsub(/\n/, " ", body)
        gsub(/[[:space:]]+/, " ", body)
        gsub(/^ +| +$/, "", body)
        affid[++nf] = ids
        afftxt[nf] = body
        p = substr(p, bp + length(body) + 2)
    }

    print "---"                       > meta
    print "title: '" tomd(title) "'"  > meta
    print "author:"                   > meta
    for (i = 1; i <= na; i++)
        print "  - '" auth[i] "'"     > meta
    print "---"                       > meta

    # ---- 3. body ----
    for (i = 1; i <= nf; i++)
        print "\\textsuperscript{" affid[i] "}" afftxt[i] "\\newline"
    print ""

    for (i = bodystart + 1; i <= n; i++) {
        line = L[i]
        if (line ~ /\\maketitle/)            continue
        if (line ~ /\\end[{]abstract[}]/)    continue
        if (line ~ /\\bibliographystyle[{]/) continue
        if (line ~ /\\bibliography[{]/)      continue
        if (line ~ /\\end[{]document[}]/)    continue
        if (line ~ /\\begin[{]abstract[}]/) { print "\\section*{Abstract}"; continue }

        # longtable repeats its header on every printed page.  Word reflows
        # instead of paginating, so the continuation block (the "... continued"
        # line and the second copy of the header) has nothing to do here --
        # pandoc otherwise emits it as ordinary data rows in the middle of the
        # table, and the duplicate \endhead leaves the table with no header at
        # all.  Keep the first header, drop the repeat and the foot markers.
        if (line ~ /\\begin[{]longtable[}]/) inlt = 1
        if (line ~ /\\end[{]longtable[}]/)   inlt = 0
        if (inlt) {
            if (skiphead) { if (line ~ /\\endhead/) skiphead = 0; continue }
            if (line ~ /\\endfirsthead/) { print "\\endhead"; skiphead = 1; continue }
            if (line ~ /\\endfoot/ || line ~ /\\endlastfoot/) continue

            # Buffer the header rows (\toprule .. \midrule) and merge them.
            if (!hdrdone && line ~ /\\toprule/) { print line; inhdr = 1; hdr = ""; continue }
            if (inhdr) {
                if (line ~ /\\midrule/ || line ~ /\\endhead/) {
                    print flatten_header(hdr)
                    print line
                    inhdr = 0
                    hdrdone = 1
                    continue
                }
                hdr = hdr " " line
                continue
            }
        } else { hdrdone = 0 }

        # Unwrap figure floats.  Inside a float LaTeX hoists \caption to the
        # bottom, which would strand a figure supplement between the main
        # image and its own legend.  Without the float, everything stays in
        # source order: image, then legend, then each supplement.
        if (line ~ /\\begin[{]figure[}]/)  { infig = 1; continue }
        if (line ~ /\\end[{]figure[}]/)    { infig = 0; continue }
        if (infig && line ~ /\\(begin|end)[{]center[}]/) continue
        if (infig) sub(/\\caption[{]/, "\\FIGCAP{", line)

        print fixrefs(line)
    }

    # citeproc appends the bibliography at the end of the document, so the
    # heading has to go here rather than where \bibliography sat.
    print ""
    print "\\section*{References}"
}
