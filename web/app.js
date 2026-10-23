// Loads Pyodide, the esperanto_nlp library and the exported model, then
// classifies the text with web/demo.py on every click.

const CLASSES = {
  kristanismo: "Christianity",
  islamo: "Islam",
  judismo: "Judaism",
};

const EXAMPLES = [
  ["Islam", "Dum la monato Ramadano la islamanoj fastas de la sunleviĝo ĝis la sunsubiro. Vespere ili kunvenas en la moskeo por preĝi kaj legi la Koranon."],
  ["Christianity", "La apostolo Paŭlo vojaĝis tra la Romia Imperio por prediki la evangelion de Jesuo Kristo kaj fondi novajn preĝejojn."],
  ["Judaism", "Ĉiun sabaton la judoj kunvenas en la sinagogo, kie la rabeno legas parton de la Torao en la hebrea lingvo."],
  ["Shared history", "Abraham estas konsiderata patro de la kredantoj. Laŭ la tradicio, Dio promesis al li grandan posteularon kaj la landon Kanaano."],
  ["x-system", "La pastro cxiutage diras la meson en la katedralo kaj benas la kredantojn."],
];

const LOW_CONFIDENCE = 0.5;

const $ = (id) => document.getElementById(id);
let analyze = null;

async function fetchOk(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url}: HTTP ${response.status}`);
  return response;
}

async function init() {
  const pyodide = await loadPyodide();
  const [library, demo, model] = await Promise.all([
    fetchOk("esperanto_nlp.zip").then((r) => r.arrayBuffer()),
    fetchOk("demo.py").then((r) => r.text()),
    fetchOk("model.json").then((r) => r.text()),
  ]);
  pyodide.unpackArchive(library, "zip");
  pyodide.FS.writeFile("demo.py", demo);
  pyodide.globals.set("model_json", model);
  analyze = pyodide.runPython(`
import json
import demo
model = demo.Model(json.loads(model_json))
del model_json
lambda text: demo.analyze(text, model)
`);
}

function el(tag, props = {}, children = []) {
  const node = Object.assign(document.createElement(tag), props);
  node.append(...children);
  return node;
}

function render(result) {
  const best = result.predicted;
  const confidence = result.probabilities[best];

  $("verdict").replaceChildren(
    "Most likely about ",
    el("strong", { textContent: CLASSES[best] }),
    ` (${Math.round(confidence * 100)}%)`,
  );

  $("bars").replaceChildren(...Object.entries(result.probabilities).map(([label, p]) =>
    el("div", { className: "bar-row" }, [
      el("div", { className: "bar-label", textContent: CLASSES[label] }, [
        el("small", { textContent: label }),
      ]),
      el("div", { className: "bar-track" }, [
        el("div", {
          className: "bar-fill",
          style: `width:${(p * 100).toFixed(1)}%;background:var(--${label})`,
        }),
      ]),
      el("div", { className: "bar-value", textContent: `${(p * 100).toFixed(1)}%` }),
    ])
  ));
  $("low-confidence").classList.toggle("hidden", confidence >= LOW_CONFIDENCE);

  $("evidence").replaceChildren(...result.evidence.map(([term]) =>
    el("span", { className: "chip", textContent: term })
  ));
  $("evidence-note").textContent = result.evidence.length
    ? `Lemmas and lemma pairs that pushed the text towards ${CLASSES[best]} the most, strongest first.`
    : "None of the words in this text pointed clearly to any class.";
  $("why").classList.toggle("hidden", result.known_features === 0);

  $("tokens").replaceChildren(...result.tokens.map((t) =>
    el("tr", { className: t.stop ? "stop" : "" }, [
      el("td", { textContent: t.text }),
      el("td", { textContent: t.lemma }),
      el("td", { textContent: t.pos }),
      el("td", { textContent: t.morph || "—" }),
    ])
  ));

  for (const id of ["result", "analysis"]) $(id).classList.remove("hidden");
}

function classify() {
  const text = $("text").value.trim();
  if (!text) {
    $("status").textContent = "Write or paste some Esperanto text first.";
    return;
  }
  const start = performance.now();
  render(JSON.parse(analyze(text)));
  $("status").textContent = `Analysed in ${Math.round(performance.now() - start)} ms.`;
}

for (const [label, text] of EXAMPLES) {
  $("examples").append(el("button", {
    className: "chip",
    textContent: label,
    onclick: () => {
      $("text").value = text;
      if (analyze) classify();
    },
  }));
}
$("text").value = EXAMPLES[0][1];
$("classify").addEventListener("click", classify);
$("text").addEventListener("keydown", (event) => {
  if (event.key === "Enter" && (event.ctrlKey || event.metaKey) && analyze) classify();
});

init()
  .then(() => {
    $("classify").disabled = false;
    $("status").textContent = "Ready. Press Classify (or Ctrl+Enter).";
    classify();
  })
  .catch((error) => {
    console.error(error);
    $("status").textContent = `Could not load the classifier: ${error.message}`;
  });
