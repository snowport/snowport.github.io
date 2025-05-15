document.addEventListener("DOMContentLoaded", function () {
  if (typeof renderMathInElement !== "undefined") {
    renderMathInElement(document.body, {
      delimiters: [
        {left: "$$", right: "$$", display: true},
        {left: "$", right: "$", display: false}
      ]
    });
  } else {
    console.warn("KaTeX renderMathInElement is not defined.");
  }
});