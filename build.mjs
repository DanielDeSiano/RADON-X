import { cp, mkdir, readdir, readFile, writeFile } from 'node:fs/promises';
import { execFileSync } from 'node:child_process';
import { join } from 'node:path';

const runtime = 'dist/runtime';
await mkdir(runtime, { recursive: true });
await cp('site', 'dist', { recursive: true });
const template = await readFile('site/index.html', 'utf8');
const pageNames = { project: 'The project', lab: 'Simulation lab', daniel: 'Daniel De Siano' };
for (const [page, title] of Object.entries(pageNames)) {
  const folder = page === 'project' ? 'dist' : join('dist', page);
  await mkdir(folder, { recursive: true });
  const html = template
    .replace(/<!-- page:(\w+) -->([\s\S]*?)<!-- endpage -->/g, (_, owner, content) => owner === page ? content : '')
    .replace('<title>RADON-X — Radon research from Ischia</title>', `<title>${title} | RADON-X</title>`)
    .replace(`data-page="${page}"`, 'aria-current="page"');
  await writeFile(join(folder, 'index.html'), html);
}
await cp('research/data', 'dist/data', { recursive: true });
await cp('research/examples/ischia_demo.json', 'dist/example.json');
await cp('node_modules/gsap/dist/gsap.min.js', 'dist/gsap.min.js');
for (const file of ['pyodide.js', 'pyodide.asm.js', 'pyodide.asm.wasm', 'python_stdlib.zip', 'pyodide-lock.json']) {
  await cp(join('node_modules/pyodide', file), join(runtime, file));
}
const lock = JSON.parse(await readFile(join(runtime, 'pyodide-lock.json'), 'utf8'));
const wheels = new Set();
function collect(name) {
  if (wheels.has(name)) return;
  wheels.add(name);
  for (const dependency of lock.packages[name].depends) collect(dependency);
}
collect('scipy');
const { createHash } = await import('node:crypto');
await Promise.all([...wheels].map(async name => {
  const entry = lock.packages[name];
  const file = join(runtime, entry.file_name);
  execFileSync('curl', ['-fsSL', '--retry', '2', '-o', file,
    'https://cdn.jsdelivr.net/pyodide/v0.27.7/full/' + entry.file_name]);
  const hash = createHash('sha256').update(await readFile(file)).digest('hex');
  if (hash !== entry.sha256) throw new Error('Invalid package checksum: ' + name);
}));
const sources = {};
for (const file of await readdir('research/radon_x', {recursive: true})) {
  if (file.endsWith('.py')) sources['radon_x/' + file] = await readFile(join('research/radon_x', file), 'utf8');
}
sources['bridge.py'] = await readFile('research/bridge.py', 'utf8');
await writeFile('dist/model.json', JSON.stringify(sources));
const data = (await readdir('research/data', {recursive: true})).filter(file => /\.(csv|json|md|txt)$/i.test(file)).sort();
await writeFile('dist/datasets.json', JSON.stringify(data));
