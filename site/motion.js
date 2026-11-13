// Bang Motion approach: one continuous world, rendered from timeline time.
const canvas = document.querySelector('#geologyCanvas');
const ctx = canvas.getContext('2d');
const control = document.querySelector('#motionControl');
const reduced = matchMedia('(prefers-reduced-motion: reduce)');
const clock = { t: 0 };
const noise = n => ((Math.sin(n * 78.233) * 43758.5453) % 1 + 1) % 1;
const strata = ['#b58a66', '#976d53', '#79523e', '#50392e'];

function horizon(x, i) {
  return 206 + i * 112 + 21 * Math.sin(x / 155 + i * .7) + 12 * Math.sin(x / 65 + i);
}

function render(t) {
  ctx.clearRect(0, 0, 960, 680);
  const sky = ctx.createLinearGradient(0, 0, 900, 260);
  sky.addColorStop(0, '#faf3e1');
  sky.addColorStop(1, '#dac7a6');
  ctx.fillStyle = sky;
  ctx.fillRect(0, 0, 960, 680);
  ctx.fillStyle = '#bb5436';
  ctx.beginPath();
  ctx.arc(755 - 18 * Math.sin(t / 3), 106, 53, 0, 2 * Math.PI);
  ctx.fill();
  strata.forEach((color, layer) => {
    ctx.beginPath();
    ctx.moveTo(0, horizon(0, layer));
    for (let x = 4; x <= 960; x += 4) ctx.lineTo(x, horizon(x, layer));
    ctx.lineTo(960, 680);
    ctx.lineTo(0, 680);
    ctx.closePath();
    ctx.fillStyle = color;
    ctx.fill();
  });
  for (let i = 0; i < 1800; i++) {
    const x = noise(i * 3 + 1) * 960;
    const y = noise(i * 3 + 2) * 680;
    ctx.fillStyle = i % 3 ? '#36251922' : '#fdf3d72c';
    ctx.fillRect(x, y, 1 + noise(i * 3 + 3) * 4, 1);
  }
  for (let i = 0; i < 60; i++) {
    const phase = Math.min(1, Math.max(0, (t - noise(i + 4) * 8) / 6));
    const x = 100 + noise(i + 90) * 750 + 25 * Math.sin(phase * 9 + i);
    const y = 625 - phase * (455 + noise(i + 6) * 62);
    ctx.globalAlpha = Math.sin(Math.PI * phase) * .9;
    ctx.fillStyle = '#ffe2a4';
    ctx.beginPath();
    ctx.arc(x, y, 2 + noise(i + 9) * 4, 0, 2 * Math.PI);
    ctx.fill();
  }
  ctx.globalAlpha = 1;
}

const timeline = gsap.timeline({ paused: true, onComplete: () => { control.textContent = 'Replay'; } });
timeline.to(clock, { t: 15, duration: 15, ease: 'none', onUpdate: () => render(clock.t) });
function applyPreference() {
  timeline.pause(0);
  render(reduced.matches ? 7 : 0);
  control.textContent = reduced.matches ? 'Play' : 'Pause';
  if (!reduced.matches) timeline.play();
}
control.addEventListener('click', () => {
  if (timeline.isActive()) {
    timeline.pause();
    control.textContent = 'Play';
  } else {
    if (timeline.progress() === 1) timeline.restart();
    else timeline.play();
    control.textContent = 'Pause';
  }
});
reduced.addEventListener('change', applyPreference);
applyPreference();
