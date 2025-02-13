function parseParagraphs(htmlString) {
  const parser = new DOMParser();
  const doc = parser.parseFromString(htmlString, 'text/html');
  const paragraphs = Array.from(doc.getElementsByTagName('p')).map(p => p.textContent);
  return paragraphs;
}
const currentPageHTML = document.documentElement.outerHTML;
const allParagraphs = parseParagraphs(currentPageHTML);
console.log(allParagraphs);
// Or even simpler, directly:
const paragraphs = Array.from(document.getElementsByTagName('p')).map(p => p.textContent);
const blob = new Blob([JSON.stringify(paragraphs, null, 2)], { type: 'application/json' });
const url = URL.createObjectURL(blob);
const link = document.createElement('a');
link.href = url;
link.download = 'paragraphs.json';
link.click();
URL.revokeObjectURL(url);
