const $ = (selector) => document.querySelector(selector);

const example = {
  schema_version: '1.0',
  site: {
    name: 'Ischia exploratory site',
    lithotype: 'tuff_unwelded',
    ra226_bq_kg: 185,
    emanation_fraction: 0.31,
    porosity: 0.42,
    dry_bulk_density_kg_m3: 1612,
    water_saturation: 0.11,
    column_depth_m: 3,
    initial_pore_accessibility: 1,
    provenance: {}
  },
  simulation: {
    cells: 30,
    duration_days: 6,
    output_step_seconds: 3600,
    average_start_day: 4.5,
    average_end_day: 5.5
  }
};

let worker;
let currentResult;
let baselineResult;
function simulate(payload) {
  return new Promise((resolve, reject) => {
    worker ??= new Worker('/worker.js');
    worker.onmessage = ({ data }) => data.error ? reject(new Error(data.error)) : resolve(data.result);
    worker.onerror = () => {
      worker.terminate();
      worker = null;
      reject(new Error('Unable to start the simulation engine. Try again.'));
    };
    worker.postMessage(payload);
  });
}

function number(value, digits = 3) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return '—';
  const n = Number(value);
  if (!Number.isFinite(n)) return '—';
  return n.toLocaleString('en-US', { maximumFractionDigits: digits });
}

function scientific(value) {
  if (value === null || value === undefined || !Number.isFinite(Number(value))) return '—';
  return Number(value).toExponential(3);
}

function toast(message, error = false) {
  const el = $('#toast');
  el.textContent = message;
  el.style.borderColor = error ? 'var(--red)' : 'var(--line-bright)';
  el.classList.add('show');
  window.clearTimeout(toast.timer);
  toast.timer = window.setTimeout(() => el.classList.remove('show'), 3600);
}

function configFromForm() {
  const averageEnd = Math.max(0.5, Number($('#duration').value) - 0.5);
  const averageStart = Math.max(0, averageEnd - Math.min(1, averageEnd));
  return {
    schema_version: '1.0',
    site: {
      name: $('#siteName').textContent || 'Local exploratory site',
      lithotype: 'tuff_unwelded',
      ra226_bq_kg: Number($('#ra226').value),
      emanation_fraction: Number($('#emanation').value),
      porosity: Number($('#porosity').value),
      dry_bulk_density_kg_m3: 1612,
      water_saturation: Number($('#saturation').value),
      column_depth_m: Number($('#depth').value),
      initial_pore_accessibility: 1,
      provenance: { interface: 'RADON-X Local Lab' }
    },
    simulation: {
      cells: Number($('#cells').value),
      duration_days: Number($('#duration').value),
      output_step_seconds: Number($('#step').value),
      average_start_day: averageStart,
      average_end_day: averageEnd
    }
  };
}

function optionPayload() {
  return { options: { base_condition: $('#boundary').value, compute_indoor: $('#computeIndoor').checked } };
}

function setForm(config) {
  const site = config.site || example.site;
  const simulation = config.simulation || example.simulation;
  $('#siteName').textContent = site.name || example.site.name;
  $('#ra226').value = site.ra226_bq_kg;
  $('#emanation').value = site.emanation_fraction;
  $('#porosity').value = site.porosity;
  $('#saturation').value = site.water_saturation;
  $('#depth').value = site.column_depth_m;
  $('#cells').value = simulation.cells;
  $('#duration').value = simulation.duration_days;
  $('#step').value = simulation.output_step_seconds;
}

function nicePath(points, width, height, pad, xMax, yMin, yMax) {
  const span = Math.max(yMax - yMin, 1e-15);
  return points.map((point, index) => {
    const x = pad.left + (point.x / Math.max(xMax, 1e-9)) * (width - pad.left - pad.right);
    const y = height - pad.bottom - ((point.y - yMin) / span) * (height - pad.top - pad.bottom);
    return `${index ? 'L' : 'M'} ${x.toFixed(2)} ${y.toFixed(2)}`;
  }).join(' ');
}

function renderFluxChart(series) {
  const svg = $('#fluxChart');
  const width = 900; const height = 300; const pad = { left: 54, right: 16, top: 19, bottom: 34 };
  if (!series || !series.length) { svg.innerHTML = '<text x="450" y="150" text-anchor="middle" fill="#64758c" font-size="12">Run a simulation to render the time series</text>'; return; }
  const values = series.map((p) => Number(p.flux_bq_m2_s)).filter(Number.isFinite);
  const max = Math.max(...values); const min = Math.min(...values);
  const spread = Math.max(max - min, Math.abs(max) * .08, 1e-7);
  const yMin = Math.max(0, min - spread * .35); const yMax = max + spread * .35;
  const points = series.map((p) => ({ x: Number(p.day), y: Number(p.flux_bq_m2_s) }));
  const path = nicePath(points, width, height, pad, Math.max(...points.map((p) => p.x)), yMin, yMax);
  const areaPath = `${path} L ${width - pad.right} ${height - pad.bottom} L ${pad.left} ${height - pad.bottom} Z`;
  const grid = [0, .5, 1].map((fraction) => {
    const y = height - pad.bottom - fraction * (height - pad.top - pad.bottom);
    const value = yMin + fraction * (yMax - yMin);
    return `<line x1="${pad.left}" y1="${y}" x2="${width - pad.right}" y2="${y}" stroke="#223249"/><text x="${pad.left - 9}" y="${y + 4}" text-anchor="end" fill="#64758c" font-size="10" font-family="Fira Code, monospace">${scientific(value)}</text>`;
  }).join('');
  const xMax = Math.max(...points.map((p) => p.x));
  svg.innerHTML = `<defs><linearGradient id="fluxFill" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#3ea5ff" stop-opacity=".25"/><stop offset="1" stop-color="#3ea5ff" stop-opacity="0"/></linearGradient></defs>${grid}<path d="${areaPath}" fill="url(#fluxFill)"/><path d="${path}" fill="none" stroke="#3ea5ff" stroke-width="2.4"/><text x="${pad.left}" y="${height - 9}" fill="#64758c" font-size="10" font-family="Fira Code, monospace">0 d</text><text x="${width - pad.right}" y="${height - 9}" text-anchor="end" fill="#64758c" font-size="10" font-family="Fira Code, monospace">${number(xMax, 1)} d</text>`;
}

function renderProfileChart(profile) {
  const svg = $('#profileChart');
  const width = 760; const height = 260; const pad = { left: 60, right: 20, top: 15, bottom: 32 };
  if (!profile || !profile.length) { svg.innerHTML = '<text x="380" y="130" text-anchor="middle" fill="#64758c" font-size="12">Profile not available</text>'; return; }
  const values = profile.map((p) => Number(p.concentration_bq_m3));
  const max = Math.max(...values, 1); const xMax = max * 1.12;
  const yMax = Math.max(...profile.map((p) => Number(p.depth_m)));
  const points = profile.map((p) => ({ x: Number(p.concentration_bq_m3), y: Number(p.depth_m) }));
  const path = points.map((point, index) => {
    const x = pad.left + (point.x / xMax) * (width - pad.left - pad.right);
    const y = pad.top + (point.y / Math.max(yMax, 1e-9)) * (height - pad.top - pad.bottom);
    return `${index ? 'L' : 'M'} ${x.toFixed(2)} ${y.toFixed(2)}`;
  }).join(' ');
  const grids = [0, .5, 1].map((f) => {
    const x = pad.left + f * (width - pad.left - pad.right);
    const val = f * xMax;
    return `<line x1="${x}" y1="${pad.top}" x2="${x}" y2="${height - pad.bottom}" stroke="#223249"/><text x="${x}" y="${height - 11}" text-anchor="middle" fill="#64758c" font-size="10" font-family="Fira Code, monospace">${number(val, 0)}</text>`;
  }).join('');
  svg.innerHTML = `${grids}<path d="${path}" fill="none" stroke="#f3ae56" stroke-width="2.3"/><text x="10" y="${pad.top + 4}" fill="#64758c" font-size="10" font-family="Fira Code, monospace">0 m</text><text x="10" y="${height - pad.bottom}" fill="#64758c" font-size="10" font-family="Fira Code, monospace">${number(yMax, 1)} m</text>`;
}

function updateWarnings(result) {
  const list = $('#warningsList');
  const warnings = result.warnings || [];
  const diagnostics = result.diagnostics || {};
  const residual = Number(diagnostics.relative_balance_residual);
  const checks = [];
  if (warnings.length) warnings.forEach((warning) => checks.push({ text: warning, cls: 'warn' }));
  else checks.push({ text: 'Configuration accepted by the typed core.', cls: '' });
  checks.push({ text: `Relative balance residual: ${scientific(residual)}`, cls: Math.abs(residual) > 1e-5 ? 'warn' : '' });
  checks.push({ text: `Minimum concentration: ${number(diagnostics.minimum_concentration_bq_m3, 2)} Bq m⁻³`, cls: '' });
  list.replaceChildren(...checks.map(({text, cls}) => {
    const item = document.createElement('li');
    item.className = cls;
    item.textContent = text;
    return item;
  }));
  $('#checkIcon').textContent = warnings.length ? '!' : '✓';
  $('#checkIcon').style.color = warnings.length ? 'var(--amber)' : 'var(--mint)';
}

function renderResult(result) {
  currentResult = result;
  $('#saveBaseline').disabled = false;
  renderComparison();
  const diagnostics = result.diagnostics || {};
  $('#surfaceFlux').textContent = scientific(result.surface_flux_bq_m2_s);
  $('#soilMean').textContent = number(result.soil_mean_bq_m3, 1);
  $('#indoorValue').textContent = number(result.indoor_bq_m3, 1);
  $('#balanceValue').textContent = scientific(diagnostics.balance_residual_bq_m2_s);
  $('#runTime').textContent = `runtime ${number(result.runtime_seconds, 3)} s`;
  $('#fluxRange').textContent = `${result.series?.length || 0} output points · ${number(result.surface_flux_bq_m2_day, 2)} Bq m⁻² day⁻¹`;
  $('#profileCount').textContent = result.column_profile?.length || 0;
  $('#referenceDepth').textContent = `${number(result.reference_depth_m, 2)} m`;
  $('#diffusionLength').textContent = `${number(diagnostics.diffusion_length_m, 3)} m`;
  $('#fourierNumber').textContent = number(diagnostics.max_cell_fourier_number, 5);
  $('#readoutTitle').textContent = result.indoor_bq_m3 == null ? 'Core run complete' : 'Indoor coupling computed';
  $('#readoutText').textContent = `Simulated period: ${result.request.config.simulation.duration_days} days. Soil mean: ${number(result.soil_mean_bq_m3, 1)} Bq m⁻³. The indoor estimate is exploratory and depends on the model assumptions.`;
  renderFluxChart(result.series);
  renderProfileChart(result.column_profile);
  updateWarnings(result);
}

async function runSimulation(event) {
  event?.preventDefault();
  const button = document.querySelector('.primary-button');
  button.disabled = true;
  button.querySelector('span').textContent = 'Running core…';
  try {
    $('#runError').textContent = '';
    renderResult(await simulate({config: configFromForm(), ...optionPayload()}));
    toast('Simulation complete — diagnostics updated.');
  } catch (error) {
    $('#runError').textContent = error.message;
    toast(error.message, true);
  } finally {
    button.disabled = false;
    button.querySelector('span').textContent = 'Run simulation';
  }
}

async function loadExample() {
  try {
    const response = await fetch('/example.json');
    setForm(response.ok ? await response.json() : example);
  } catch { setForm(example); }
}

$('#simulationForm').addEventListener('submit', runSimulation);
$('#resetButton').addEventListener('click', async () => {
  await loadExample();
  $('#boundary').value = 'sealed';
  $('#computeIndoor').checked = true;
  toast('Example configuration restored. Run again to update the results.');
});
loadExample();
renderFluxChart([]);
renderProfileChart([]);

function renderComparison() {
  if (!baselineResult) return;
  const metrics = [
    ['Surface flux (Bq m⁻² s⁻¹)', 'surface_flux_bq_m2_s', scientific],
    ['Soil mean (Bq m⁻³)', 'soil_mean_bq_m3', number],
    ['Indoor estimate (Bq m⁻³)', 'indoor_bq_m3', number]
  ];
  $('#comparisonRows').replaceChildren(...metrics.map(([label, key, format]) => {
    const before = baselineResult[key];
    const after = currentResult[key];
    const change = Number.isFinite(before) && before !== 0 && Number.isFinite(after)
      ? new Intl.NumberFormat('en', {style: 'percent', signDisplay: 'exceptZero', maximumFractionDigits: 1}).format((after - before) / before)
      : '—';
    const row = document.createElement('tr');
    [label, format(before), format(after), change].forEach((value, i) => {
      const cell = document.createElement(i === 0 ? 'th' : 'td');
      if (i === 0) cell.scope = 'row';
      cell.textContent = value;
      row.append(cell);
    });
    return row;
  }));
  $('#comparisonTable').hidden = false;
  $('#comparisonStatus').textContent = baselineResult === currentResult
    ? 'Baseline saved for this session. Change an input and run again.'
    : 'Changes are relative to the saved baseline. Compare runs with the same duration to isolate the effect of one parameter.';
}

$('#saveBaseline').addEventListener('click', () => {
  baselineResult = currentResult;
  renderComparison();
});

async function loadArchive() {
  const status = $('#datasetStatus');
  try {
    const response = await fetch('/datasets.json');
    if (!response.ok) throw new Error();
    const files = await response.json();
    const filter = $('#datasetFilter');
    function render() {
      const query = filter.value.toLowerCase().trim();
      const matches = files.filter(file => file.toLowerCase().includes(query));
      status.textContent = `${matches.length} files`;
      $('#datasetList').replaceChildren(...matches.map(file => {
        const li = document.createElement('li');
        const link = document.createElement('a');
        link.href = '/data/' + file.split('/').map(encodeURIComponent).join('/');
        link.download = '';
        link.textContent = file;
        li.append(link);
        return li;
      }));
    }
    filter.addEventListener('input', render);
    render();
  } catch {
    status.textContent = 'Data archive unavailable. Reload to try again.';
  }
}
loadArchive();
