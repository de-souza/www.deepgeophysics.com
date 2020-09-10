const katex_codeblocks = (() => {
  for (const code of document.querySelectorAll("div > pre > .language-tex")) {
    const fragment = document.createDocumentFragment();
    katex.render(code.innerText, fragment, { displayMode: true });
    code.parentNode.parentNode.replaceWith(fragment);
  };
})();
