// Template for article
// Technical formatting aligned with NDT template

// ── Shared helper functions ────────────────────────────────────────────────

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

// ── Wide-block helpers ─────────────────────────────────────────────────────

#let wide-block(body, extra-margin: 0.75in) = {
  block(
    inset: (
      left: -extra-margin,
      right: -extra-margin,
      top: 0em,
      bottom: 0em,
    ),
    width: auto,
  )[body]
}

#let wide-table(table-content, extra-margin: 0.75in) = wide-block(
  table-content,
  extra-margin: extra-margin,
)

#let wide-figure(figure-content, extra-margin: 0.75in) = wide-block(
  figure-content,
  extra-margin: extra-margin,
)

// ── Main template ──────────────────────────────────────────────────────────

#let essay-template(
  title: none,
  short_title: none,
  subtitle: none,
  // author/authors are aliases
  author: none,
  authors: none,
  // affiliation/affiliations are aliases
  affiliation: none,
  affiliations: none,
  // correspondence is alias for corresponding_email
  corresponding_email: none,
  correspondence: none,
  contributor_notes: none,
  abstract: none,
  date: none,
  doi: none,
  numbered_headings: false,
  doc,
) = {

  // Resolve aliases
  let _authors = if authors != none { authors } else { author }
  let _affiliations = if affiliations != none { affiliations } else { affiliation }
  let _correspondence = if correspondence != none { correspondence } else { corresponding_email }

  // Page setup
  set page(
    paper: "a4",
    margin: (
      top: 2.5cm,
      bottom: 2.5cm,
      left: 2.5cm,
      right: 2.5cm,
    ),
    header: context {
      let page-num = counter(page).get().first()
      if page-num > 1 {
        let header-title = if short_title != none {
          short_title
        } else if title != none {
          title
        } else {
          ""
        }
        grid(
          columns: (1fr, auto),
          align(left)[#text(size: 9pt, style: "italic")[#header-title]],
          align(right)[#text(size: 9pt)[#page-num]],
        )
      }
    },
    numbering: "1",
  )

  // Text setup
  set text(
    font: "New Computer Modern",
    size: 11pt,
    lang: "en",
    hyphenate: true,
  )

  // Paragraph setup
  set par(
    justify: true,
    leading: 0.6em,
    spacing: 0.75em,
    first-line-indent: 1.2em,
  )

  // Heading numbering
  if numbered_headings {
    set heading(numbering: "1.")
  } else {
    set heading(numbering: none)
  }

  // Math equation styling
  set math.equation(numbering: "(1)")

  // Heading styles
  show heading.where(level: 1): it => {
    set text(size: 14pt, weight: "bold")
    set block(above: 1.8em, below: 1em)
    align(left)[#it]
  }

  show heading.where(level: 2): it => {
    set text(size: 12pt, weight: "semibold")
    set block(above: 1.4em, below: 0.7em)
    it
  }

  show heading.where(level: 3): it => {
    set text(size: 11pt, weight: "semibold", style: "italic")
    set block(above: 1em, below: 0.5em)
    it
  }

  // Quote/emphasis styling
  show emph: it => text(style: "italic", it.body)
  show strong: it => text(weight: "bold", it.body)

  // Link styling
  show link: it => text(fill: rgb("#0066cc"), it)

  // Code/technical term styling
  show raw.where(block: false): it => box(
    fill: luma(240),
    inset: (x: 3pt, y: 0pt),
    outset: (y: 3pt),
    radius: 2pt,
    text(font: ("Courier New", "DejaVu Sans Mono", "Consolas"), size: 0.9em, it)
  )

  // Table styling
  show table: it => {
    set text(size: 9.5pt)
    set par(
      first-line-indent: 0pt,
      leading: 0.55em,
    )
    block(above: 1.2em, below: 1.2em)[
      #it
    ]
  }

  // Reset paragraph indent for first paragraph after each heading
  show heading: it => {
    it
    par(first-line-indent: 0em)[#text(size: 0pt)[]]
  }

  // ── Title page ───────────────────────────────────────────────────────────
  if title != none {
    align(center)[
      #v(1.2cm)
      #text(size: 18pt, weight: "bold")[#title]

      #if contributor_notes != none {
        v(0.7em)
        contributor_notes
      }

      #v(1.4em)

      #if _authors != none {
        text(size: 12pt)[#_authors]
      }

      #if _affiliations != none {
        v(0.3em)
        text(size: 11pt, style: "italic")[#_affiliations]
      }

      #if _correspondence != none {
        v(0.4em)
        text(size: 10pt)[Correspondence: #_correspondence]
      }

      #if short_title != none {
        v(0.5em)
        text(size: 9pt, style: "italic")[Running title: #short_title]
      }

      #if date != none {
        v(0.5em)
        text(size: 10pt)[#date]
      }

      #if doi != none {
        v(0.4em)
        text(size: 9pt)[DOI: #doi]
      }
    ]

    v(1.2cm)
  }

  // ── Abstract ─────────────────────────────────────────────────────────────
  if abstract != none {
    abstract-block(abstract)
  }

  // ── Document body ─────────────────────────────────────────────────────────
  doc
}
