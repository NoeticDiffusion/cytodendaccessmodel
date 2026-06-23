// Template for eLife-style Typst articles
// Extracted from the original example so manuscripts can stay focused on content.

#let note-line(label, body) = par(first-line-indent: 0pt)[
  #emph(label)#text(": ")#body
]

#let supplement-box(title, body) = block(
  inset: 10pt,
  stroke: 0.5pt + black,
  above: 8pt,
  below: 8pt,
)[
  #par(first-line-indent: 0pt)[#strong(title)]
  #body
]

#let abstract-block(body) = block(
  stroke: (
    top: 0.8pt + black,
    bottom: 0.8pt + black,
  ),
  inset: (
    top: 8pt,
    bottom: 8pt,
  ),
  above: 12pt,
  below: 14pt,
)[
  #grid(
    columns: (auto, 1fr),
    column-gutter: 10pt,
    align: (left, top),
    [#par(first-line-indent: 0pt)[#strong[Abstract]]],
    [
      #set par(first-line-indent: 0pt)
      #body
    ],
  )
]

#let clean-table(
  columns: auto,
  header: (),
  align: left,
  inset: 5pt,
  midrule: 0.5pt + black,
  ..body,
) = context {
  set text(size: 8.2pt)
  table(
    columns: columns,
    inset: inset,
    align: align,
    stroke: none,
    table.header(..header),
    table.hline(y: 1, stroke: midrule),
    ..body,
  )
}

#let elife-template(
  title: none,
  authors: none,
  affiliations: none,
  correspondence: none,
  contributor_notes: none,
  abstract: none,
  doc,
) = {
  set page(
    paper: "a4",
    margin: (
      left: 5cm,
      right: 2cm,
      top: 2.2cm,
      bottom: 2.2cm,
    ),
  )

  set text(
    font: "Times New Roman",
    size: 9pt,
  )

  set par(
    justify: true,
    first-line-indent: 1.5em,
    leading: 0.72em,
  )

  set heading(numbering: none)
  set math.equation(numbering: "(1)")

  if title != none {
    align(center)[
      #text(size: 16pt, weight: "bold")[#title]
    ]

    if contributor_notes != none {
      set text(size: 8pt)
      contributor_notes
      set text(size: 9pt)
    }

    v(10pt)
  }

  if authors != none {
    par(first-line-indent: 0pt)[#authors]
  }

  if affiliations != none {
    affiliations
  }

  if correspondence != none {
    set text(size: 8pt)
    par(first-line-indent: 0pt)[#strong[For correspondence:] #correspondence]
    set text(size: 9pt)
  }

  if abstract != none {
    abstract-block(abstract)
  }

  doc
}
