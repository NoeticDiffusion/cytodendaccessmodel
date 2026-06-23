// Template for article
// Technical formatting aligned with NDT template

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

#let essay-template(
  title: none,
  short_title: none,
  subtitle: none,
  author: none,
  affiliation: none,
  corresponding_email: none,
  date: none,
  doi: none,
  doc,
  ) = {
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
          "Noetic Diffusion Theory"
        }
        align(right)[
          #text(size: 9pt, style: "italic")[#header-title]
          #h(1fr)
          #page-num
        ]
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

  // Enable chapter numbering for level-1 headings only
  set heading(numbering: "1.")
  
  // Heading styles
  show heading.where(level: 1): it => {
    set text(size: 18pt, weight: "bold")
    set block(above: 1.8em, below: 1.2em)
    align(left)[
      #it
    ]
  }
  
  show heading.where(level: 2): it => {
    set text(size: 13pt, weight: "semibold")
    set block(above: 1.4em, below: 0.8em)
    it
  }
  
  show heading.where(level: 3): it => {
    set text(size: 11pt, weight: "semibold", style: "italic")
    set block(above: 1em, below: 0.5em)
    it
  }
  
  // Math equation styling
  set math.equation(numbering: "(1)")
  
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

  // Title page
  if title != none {
    set align(center)
    v(2cm)
    
    text(size: 24pt, weight: "bold")[#title]
    
    if subtitle != none {
      v(0.8em)
      text(size: 16pt, style: "italic")[#subtitle]
    }
    
    v(2em)
    
    if author != none {
      text(size: 12pt)[#author]
    }
    
    if affiliation != none {
      v(0.3em)
      text(size: 11pt, style: "italic")[#affiliation]
    }
    
    if corresponding_email != none {
      v(0.5em)
      text(size: 10pt)[Correspondence: #corresponding_email]
    }

    if short_title != none {
      v(0.8em)
      text(size: 10pt)[Short title: #short_title]
    }

    v(1em)
    
    if date != none {
      text(size: 10pt)[#date]
    }
    
    v(1.2cm)
    
    if doi != none {
      v(0.8em)
      text(size: 9pt)[DOI: #doi]
    }
    
    pagebreak()
  }
  
  // Reset paragraph indent for first paragraph after headings
  show heading: it => {
    it
    par(first-line-indent: 0em)[#text(size: 0pt)[]]
  }
  
  // Document body
  doc
}

