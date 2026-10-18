// Matplotlib exports: selectable text and vector paths, never screenshots.
#let chart(name, language, width: 140mm) = image(
  "figures/" + name + "-" + language + ".pdf", width: width,
)
