// Native Typst shapes and text; the same object data generates draw.io.
#let scenes = json("figure-data.json")
#let ink = rgb("#252930")
#let muted = rgb("#657180")
#let blue = rgb("#32649A")
#let mm = 1mm
#let pt = 1pt

#let illustration(name, language, width: 278) = {
  let scene = scenes.at(name)
  let b = scene.bounds
  let scale = width / (b.at(2) - b.at(0))
  let pt-at(p) = ((p.at(0) - b.at(0)) * scale * mm, (p.at(1) - b.at(1)) * scale * mm)
  let height = (b.at(3) - b.at(1)) * scale
  box(width: width * mm, height: height * mm)[
    #for n in scene.nodes {
      let p = pt-at((n.x - n.w/2, n.y - n.h/2))
      place(top + left, dx: p.at(0), dy: p.at(1))[
        #set par(leading: 1.5pt)
        #rect(
          width: n.w * scale * mm, height: n.h * scale * mm,
          fill: if n.role == "text" { none } else { rgb("#" + n.fill) },
          stroke: if n.role == "text" { none } else {
            (paint: rgb("#" + n.stroke), thickness: 0.8pt,
             dash: if n.role == "reject" { "dashed" } else { none })
          },
          inset: 0pt,
        )[
          #align(center + horizon)[#stack(dir: ttb, spacing: 3pt,
            ..n.at(language).split("\n").map(s => text(size: n.font * scale / 10 * pt, fill: ink, s)))]
        ]
      ]
    }
    #for e in scene.edges {
      let c = if e.kind == "feedback" or e.kind == "bidirectional" { blue } else { muted }
      let st = (paint: c, thickness: 1pt, dash: if e.kind == "feedback" { "dashed" } else { none })
      for i in range(e.points.len() - 1) {
        place(top + left)[#line(start: pt-at(e.points.at(i)), end: pt-at(e.points.at(i+1)), stroke: st)]
      }
      for tri in e.arrows {
        let ox = calc.min(..tri.map(p => p.at(0)))
        let oy = calc.min(..tri.map(p => p.at(1)))
        let origin = pt-at((ox, oy))
        place(top + left, dx: origin.at(0), dy: origin.at(1))[
          #polygon(..tri.map(p => ((p.at(0)-ox)*scale*mm, (p.at(1)-oy)*scale*mm)), fill: c, stroke: none)
        ]
      }
    }
    // Keep the labels on scale and feedback paths clear of the line.
    #for n in scene.nodes.filter(n => ("feedbacklabel", "ws", "as").contains(n.id)) {
      let p = pt-at((n.x - n.w/2, n.y - n.h/2))
      place(top + left, dx: p.at(0), dy: p.at(1))[
        #rect(width: n.w*scale*mm, height: n.h*scale*mm, fill: white, stroke: none, inset: 0pt)[
          #align(center+horizon)[#text(n.font*scale/10*pt, fill: muted, n.at(language))]
        ]
      ]
    }
  ]
}
