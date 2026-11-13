let runtime;

async function initialize() {
  importScripts('./runtime/pyodide.js');
  const python = await loadPyodide({indexURL: './runtime/', packageCacheDir: './runtime/'});
  await python.loadPackage('scipy');
  const response = await fetch('./model.json');
  if (!response.ok) throw new Error('Model files unavailable');
  for (const [name, source] of Object.entries(await response.json())) {
    const file = '/model/' + name;
    const separator = file.lastIndexOf('/');
    if (separator >= 0) python.FS.mkdirTree(file.slice(0, separator));
    python.FS.writeFile(file, source);
  }
  await python.runPythonAsync("import json, sys\nsys.path.insert(0, '/model')\nfrom bridge import simulate_payload");
  return python;
}

self.onmessage = async ({data}) => {
  try {
    runtime ??= initialize();
    const python = await runtime;
    python.globals.set('request_json', JSON.stringify(data));
    const result = python.runPython('json.dumps(simulate_payload(json.loads(request_json)), allow_nan=False)');
    self.postMessage({result: JSON.parse(result)});
  } catch (error) {
    console.error('RADON-X worker:', error, error.stack);
    self.postMessage({error: error.message || 'The scientific engine could not complete this run.'});
  }
};
