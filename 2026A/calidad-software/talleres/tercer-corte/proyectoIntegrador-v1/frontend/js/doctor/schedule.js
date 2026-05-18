import { get } from '../utils/api.js';
import { requireAuth } from '../utils/auth.js';
import { renderNavbar } from '../utils/navbar.js';

requireAuth('DOCTOR');

const DAYS = [
  { short: 'Lun', key: 'MONDAY' },
  { short: 'Mar', key: 'TUESDAY' },
  { short: 'Mié', key: 'WEDNESDAY' },
  { short: 'Jue', key: 'THURSDAY' },
  { short: 'Vie', key: 'FRIDAY' },
  { short: 'Sáb', key: 'SATURDAY' },
  { short: 'Dom', key: 'SUNDAY' },
];
const HOURS = [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19];

function parseDateTime(dt) {
  if (Array.isArray(dt)) { const [y, mo, d, h = 0, mi = 0] = dt; return new Date(y, mo - 1, d, h, mi); }
  return new Date(dt);
}

function normalizeTime(t) {
  if (!t) return '';
  if (Array.isArray(t)) {
    const [h, m = 0] = t;
    return `${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}`;
  }
  return String(t).slice(0, 5);
}

function getMonday(date) {
  const d = new Date(date);
  const day = d.getDay();
  const diff = day === 0 ? -6 : 1 - day;
  d.setDate(d.getDate() + diff);
  d.setHours(0, 0, 0, 0);
  return d;
}

function dateToISO(d) {
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`;
}

function addDays(d, n) {
  const r = new Date(d);
  r.setDate(r.getDate() + n);
  return r;
}

let weekStart = getMonday(new Date());
let scheduleConfig = []; // [{ dayOfWeek, startTime, endTime, active }]
let approvedLeaves = []; // [{ startDate, endDate, type, reason }]

function parseLeaveDate(d) {
  if (Array.isArray(d)) {
    const [y, mo, da] = d;
    return new Date(y, mo - 1, da);
  }
  return new Date(d + 'T00:00:00');
}

function isOnLeave(date) {
  for (const l of approvedLeaves) {
    const start = parseLeaveDate(l.startDate);
    const end   = parseLeaveDate(l.endDate);
    const d = new Date(date.getFullYear(), date.getMonth(), date.getDate());
    if (d >= start && d <= end) return l;
  }
  return null;
}

async function loadSchedule() {
  const spinner = document.getElementById('spinner-area');
  const area    = document.getElementById('schedule-area');
  spinner.classList.remove('d-none');
  area.classList.add('d-none');

  const weekEnd = addDays(weekStart, 6);
  document.getElementById('week-label').textContent =
    `${weekStart.toLocaleDateString('es-CO', { day: 'numeric', month: 'short' })} – ` +
    `${weekEnd.toLocaleDateString('es-CO', { day: 'numeric', month: 'short', year: 'numeric' })}`;

  const [appointments, config, leaves] = await Promise.all([
    get(`/doctor/schedule?weekStart=${dateToISO(weekStart)}`).catch(() => []),
    get('/doctor/schedule/config').catch(() => []),
    get('/doctor/leaves/approved').catch(() => []),
  ]);
  scheduleConfig = config || [];
  approvedLeaves = leaves || [];

  spinner.classList.add('d-none');
  area.classList.remove('d-none');

  buildGrid(appointments || []);
}

function configForDay(dayKey) {
  const entry = scheduleConfig.find(s => s.dayOfWeek === dayKey);
  if (!entry || !entry.active) return null;
  return {
    startH: Number(normalizeTime(entry.startTime).split(':')[0] || 0),
    endH:   Number(normalizeTime(entry.endTime).split(':')[0] || 0),
  };
}

function buildGrid(appointments) {
  // Header
  const header = document.getElementById('schedule-header');
  const weekDays = DAYS.map((_, i) => addDays(weekStart, i));
  header.innerHTML =
    '<th class="hour-col">Hora</th>' +
    weekDays.map((d, i) => `<th>${DAYS[i].short}<br>
      <small class="fw-normal text-muted">${d.toLocaleDateString('es-CO', { day: 'numeric', month: 'short' })}</small>
    </th>`).join('');

  // Group appointments by [dayIndex][hour]
  const grid = {};
  for (const a of appointments) {
    const dt  = parseDateTime(a.appointmentDateTime);
    const dayOffset = Math.round((new Date(dt.getFullYear(), dt.getMonth(), dt.getDate()) - weekStart) / 86400000);
    if (dayOffset < 0 || dayOffset > 6) continue;
    const hour = dt.getHours();
    if (!grid[dayOffset]) grid[dayOffset] = {};
    if (!grid[dayOffset][hour]) grid[dayOffset][hour] = [];
    grid[dayOffset][hour].push(a);
  }

  // Build body
  const tbody = document.getElementById('schedule-body');
  tbody.innerHTML = HOURS.map(h => `
    <tr>
      <td class="hour-col">${String(h).padStart(2, '0')}:00</td>
      ${weekDays.map((dayDate, di) => {
        const cfg   = configForDay(DAYS[di].key);
        const appts = grid[di]?.[h] || [];
        const leave = isOnLeave(dayDate);

        if (!cfg) {
          // Rest day — no working hours defined or inactive
          return `<td class="cell-rest"></td>`;
        }

        const inRange = h >= cfg.startH && h < cfg.endH;
        if (!inRange) {
          // Outside working hours
          return `<td></td>`;
        }

        if (leave) {
          // Approved leave — overrides working hours
          const title = `Permiso aprobado${leave.type ? ' · ' + leave.type : ''}${leave.reason ? ': ' + leave.reason : ''}`;
          return `<td class="cell-leave" title="${title}"></td>`;
        }

        const cellClass = appts.length === 0 ? 'cell-available' : '';
        return `<td class="${cellClass}">${appts.map(a => {
          const dt = parseDateTime(a.appointmentDateTime);
          const time = dt.toLocaleTimeString('es-CO', { hour: '2-digit', minute: '2-digit' });
          const cls = a.status.toLowerCase();
          return `<div class="appt-chip status-${cls}" title="${a.patientFullName} – ${a.reason || ''}">
            <strong>${time}</strong> ${a.patientFullName}
          </div>`;
        }).join('')}</td>`;
      }).join('')}
    </tr>`).join('');
}

document.addEventListener('DOMContentLoaded', async () => {
  renderNavbar('schedule');

  const params = new URLSearchParams(window.location.search);
  if (params.get('weekStart')) {
    weekStart = getMonday(new Date(params.get('weekStart') + 'T00:00:00'));
  }

  await loadSchedule();

  document.getElementById('btn-prev').addEventListener('click', async () => {
    weekStart = addDays(weekStart, -7);
    history.replaceState(null, '', `?weekStart=${dateToISO(weekStart)}`);
    await loadSchedule();
  });

  document.getElementById('btn-next').addEventListener('click', async () => {
    weekStart = addDays(weekStart, 7);
    history.replaceState(null, '', `?weekStart=${dateToISO(weekStart)}`);
    await loadSchedule();
  });
});
