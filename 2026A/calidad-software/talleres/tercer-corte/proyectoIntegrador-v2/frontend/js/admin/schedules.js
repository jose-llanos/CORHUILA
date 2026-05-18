import { get, put } from '../utils/api.js';
import { requireAuth } from '../utils/auth.js';
import { renderNavbar } from '../utils/navbar.js';
import { showToast, showConfirm } from '../utils/toast.js';

requireAuth('ADMIN');

const WEEKDAYS = [
  { key: 'MONDAY',    short: 'Lun', label: 'Lunes' },
  { key: 'TUESDAY',   short: 'Mar', label: 'Martes' },
  { key: 'WEDNESDAY', short: 'Mié', label: 'Miércoles' },
  { key: 'THURSDAY',  short: 'Jue', label: 'Jueves' },
  { key: 'FRIDAY',    short: 'Vie', label: 'Viernes' },
  { key: 'SATURDAY',  short: 'Sáb', label: 'Sábado' },
  { key: 'SUNDAY',    short: 'Dom', label: 'Domingo' },
];

const HOURS = [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19];
const DEFAULT_START = '08:00';
const DEFAULT_END   = '17:00';

let doctors         = [];
let currentDoctorId = null;
let currentSchedule = [];
let editState       = [];
let pristineSnapshot = '';
let approvedLeaves  = [];

const LEAVE_TYPE_LABELS = {
  VACATION: 'Vacaciones',
  SICK: 'Incapacidad',
  PERSONAL: 'Personal',
  TRAINING: 'Capacitación',
  OTHER: 'Otro',
};

const $ = (id) => document.getElementById(id);

// ─── helpers ───────────────────────────────────────────────────────────────
function normalizeTime(t) {
  if (!t) return '';
  if (Array.isArray(t)) {
    const [h, m = 0] = t;
    return `${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}`;
  }
  return String(t).slice(0, 5);
}

function buildEditState(schedule) {
  return WEEKDAYS.map(wd => {
    const found = (schedule || []).find(s => s.dayOfWeek === wd.key);
    return {
      weekDay:   wd.key,
      startTime: found ? normalizeTime(found.startTime) : DEFAULT_START,
      endTime:   found ? normalizeTime(found.endTime)   : DEFAULT_END,
      active:    found ? !!found.active : (wd.key !== 'SATURDAY' && wd.key !== 'SUNDAY'),
    };
  });
}

function snapshot() {
  return JSON.stringify(editState);
}

function dirty() {
  return snapshot() !== pristineSnapshot;
}

function initials(d) {
  return `${(d.firstName || '?')[0]}${(d.lastName || '?')[0]}`.toUpperCase();
}

function hoursOf(row) {
  if (!row.active) return 0;
  const a = Number(row.startTime.split(':')[0] || 0);
  const b = Number(row.endTime.split(':')[0]   || 0);
  const am = Number(row.startTime.split(':')[1] || 0);
  const bm = Number(row.endTime.split(':')[1]   || 0);
  return Math.max(0, (b + bm / 60) - (a + am / 60));
}

// ─── renderers ─────────────────────────────────────────────────────────────
function renderSummary() {
  const working = editState.filter(r => r.active).length;
  const resting = WEEKDAYS.length - working;
  const totalHrs = editState.reduce((acc, r) => acc + hoursOf(r), 0);
  $('summary-working').textContent = working;
  $('summary-resting').textContent = resting;
  $('summary-hours').textContent   = totalHrs.toFixed(1).replace('.0', '');
  $('unsaved-pill').classList.toggle('d-none', !dirty());
}

function renderDayCards() {
  const container = $('day-cards');
  container.innerHTML = editState.map((row, idx) => {
    const wd = WEEKDAYS[idx];
    return `
      <div class="day-card ${row.active ? 'working' : 'resting'}" data-day="${idx}">
        <div class="day-title">
          <span>${wd.label}</span>
          <span class="day-badge">${row.active ? 'Laboral' : 'Descanso'}</span>
        </div>
        <div class="form-check form-switch mb-1">
          <input class="form-check-input js-active" type="checkbox" id="sw-${idx}" ${row.active ? 'checked' : ''}>
          <label class="form-check-label small" for="sw-${idx}">
            ${row.active ? 'Día laboral' : 'Día de descanso'}
          </label>
        </div>
        <div class="time-row">
          <div>
            <label class="form-label small mb-1 text-muted">Inicio</label>
            <input type="time" class="form-control form-control-sm js-start"
                   value="${row.startTime}" ${row.active ? '' : 'disabled'}>
          </div>
          <div>
            <label class="form-label small mb-1 text-muted">Fin</label>
            <input type="time" class="form-control form-control-sm js-end"
                   value="${row.endTime}" ${row.active ? '' : 'disabled'}>
          </div>
        </div>
      </div>`;
  }).join('');

  container.querySelectorAll('[data-day]').forEach(card => {
    const idx = Number(card.dataset.day);
    card.querySelector('.js-active').addEventListener('change', (e) => {
      editState[idx].active = e.target.checked;
      renderAll();
    });
    card.querySelector('.js-start').addEventListener('change', (e) => {
      editState[idx].startTime = e.target.value;
      renderAll();
    });
    card.querySelector('.js-end').addEventListener('change', (e) => {
      editState[idx].endTime = e.target.value;
      renderAll();
    });
  });
}

function renderGrid() {
  const header = $('schedule-header');
  header.innerHTML =
    '<th class="hour-col">Hora</th>' +
    WEEKDAYS.map(wd => `<th>${wd.short}<br><small class="fw-normal text-muted">${wd.label}</small></th>`).join('');

  const body = $('schedule-body');
  body.innerHTML = HOURS.map(h => `
    <tr>
      <td class="hour-col">${String(h).padStart(2, '0')}:00</td>
      ${editState.map(row => {
        if (!row.active) return `<td class="cell-rest"></td>`;
        const startH = Number(row.startTime.split(':')[0] || 0);
        const endH   = Number(row.endTime.split(':')[0]   || 0);
        const inRange = h >= startH && h < endH;
        return `<td class="${inRange ? 'cell-available' : ''}"></td>`;
      }).join('')}
    </tr>`).join('');
}

function renderAll() {
  renderDayCards();
  renderGrid();
  renderSummary();
}

// ─── data ──────────────────────────────────────────────────────────────────
async function loadDoctors() {
  const data = await get('/admin/doctors').catch(() => []);
  doctors = (data || []).filter(d => d.active);
  const sel = $('doctor-select');
  sel.innerHTML = '<option value="">— Selecciona —</option>' +
    doctors.map(d => `<option value="${d.id}">${d.firstName} ${d.lastName} – ${d.specialtyName || 'Sin especialidad'}</option>`).join('');
}

function parseLeaveDate(d) {
  if (Array.isArray(d)) {
    const [y, mo, da] = d;
    return new Date(y, mo - 1, da);
  }
  return new Date(d + 'T00:00:00');
}

function formatDate(d) {
  return parseLeaveDate(d).toLocaleDateString('es-CO', {
    day: 'numeric', month: 'short', year: 'numeric',
  });
}

function renderLeaves() {
  const today = new Date();
  today.setHours(0, 0, 0, 0);

  const upcoming = approvedLeaves
    .filter(l => parseLeaveDate(l.endDate) >= today)
    .sort((a, b) => parseLeaveDate(a.startDate) - parseLeaveDate(b.startDate));

  const el = $('leaves-area');
  if (!upcoming.length) {
    el.innerHTML = '<div class="text-muted small">Este médico no tiene permisos aprobados pendientes o vigentes.</div>';
    return;
  }

  el.innerHTML = upcoming.map(l => {
    const start = parseLeaveDate(l.startDate);
    const end   = parseLeaveDate(l.endDate);
    const isCurrent = today >= start && today <= end;
    const range = start.getTime() === end.getTime()
      ? formatDate(l.startDate)
      : `${formatDate(l.startDate)} – ${formatDate(l.endDate)}`;
    return `
      <div class="leave-item">
        <div>
          <div class="fw-semibold">
            ${range}
            ${isCurrent ? '<span class="badge bg-warning text-dark ms-2">En curso</span>' : ''}
          </div>
          <div class="small text-muted">${l.reason || 'Sin motivo registrado'}</div>
        </div>
        <span class="leave-type">${LEAVE_TYPE_LABELS[l.type] || l.type || 'Permiso'}</span>
      </div>`;
  }).join('');
}

async function loadSchedule(doctorId) {
  $('spinner-area').classList.remove('d-none');
  $('schedule-area').classList.add('d-none');
  $('empty-state').classList.add('d-none');
  $('form-alert').classList.add('d-none');

  const [sched, leaves] = await Promise.all([
    get(`/admin/doctors/${doctorId}/schedule`).catch(() => []),
    get(`/admin/doctors/${doctorId}/leaves/approved`).catch(() => []),
  ]);
  currentSchedule = sched || [];
  approvedLeaves  = leaves || [];
  editState = buildEditState(currentSchedule);
  pristineSnapshot = snapshot();

  $('spinner-area').classList.add('d-none');
  $('schedule-area').classList.remove('d-none');

  const doc = doctors.find(d => d.id === doctorId);
  if (doc) {
    $('doctor-meta-wrap').style.display = '';
    $('doctor-avatar').textContent = initials(doc);
    $('doctor-name').textContent   = `${doc.firstName} ${doc.lastName}`;
    $('doctor-extra').textContent  = `${doc.specialtyName || 'Sin especialidad'} · Lic. ${doc.medicalLicense}`;
  }

  renderAll();
  renderLeaves();
}

// ─── validation + save ─────────────────────────────────────────────────────
function validate() {
  for (const row of editState) {
    if (!row.active) continue;
    if (!row.startTime || !row.endTime) {
      return `Define hora de inicio y fin para ${WEEKDAYS.find(w => w.key === row.weekDay).label}.`;
    }
    if (row.startTime >= row.endTime) {
      return `En ${WEEKDAYS.find(w => w.key === row.weekDay).label}: la hora de inicio debe ser menor a la de fin.`;
    }
  }
  return null;
}

function showError(msg) {
  const el = $('form-alert');
  el.textContent = msg;
  el.classList.remove('d-none');
}

async function save() {
  const err = validate();
  if (err) { showError(err); return; }
  $('form-alert').classList.add('d-none');

  const payload = editState.map(row => ({
    weekDay:   row.weekDay,
    startTime: row.startTime.length === 5 ? row.startTime + ':00' : row.startTime,
    endTime:   row.endTime.length === 5 ? row.endTime + ':00' : row.endTime,
    active:    row.active,
  }));

  const btn = $('btn-save');
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner-border spinner-border-sm me-1"></span>Guardando…';
  try {
    currentSchedule = await put(`/admin/doctors/${currentDoctorId}/schedule`, payload);
    editState = buildEditState(currentSchedule || []);
    pristineSnapshot = snapshot();
    renderAll();
    showToast('Horario actualizado correctamente.', 'success');
  } catch (e) {
    showError(e.message || 'No se pudo guardar el horario.');
  } finally {
    btn.disabled = false;
    btn.innerHTML = '<i class="bi bi-floppy me-1"></i>Guardar cambios';
  }
}

// ─── quick actions ────────────────────────────────────────────────────────
function applyMondayToAll() {
  const monday = editState[0];
  if (!monday.active) {
    showToast('Lunes está marcado como descanso, primero actívalo.', 'warning');
    return;
  }
  for (let i = 1; i < editState.length; i++) {
    if (editState[i].active) {
      editState[i].startTime = monday.startTime;
      editState[i].endTime   = monday.endTime;
    }
  }
  renderAll();
}

function enableAll() {
  editState.forEach(r => { r.active = true; });
  renderAll();
}

function weekendRest() {
  editState[5].active = false; // Saturday
  editState[6].active = false; // Sunday
  renderAll();
}

// ─── boot ──────────────────────────────────────────────────────────────────
async function handleDoctorChange(newId) {
  if (currentDoctorId && dirty()) {
    const ok = await showConfirm({
      title: 'Cambios sin guardar',
      message: 'Tienes cambios sin guardar en este horario. ¿Deseas descartarlos?',
      confirmText: 'Sí, descartar',
      cancelText: 'Seguir editando',
      variant: 'danger',
    });
    if (!ok) {
      $('doctor-select').value = currentDoctorId;
      return;
    }
  }

  currentDoctorId = newId || null;
  if (currentDoctorId) {
    await loadSchedule(currentDoctorId);
  } else {
    $('schedule-area').classList.add('d-none');
    $('empty-state').classList.remove('d-none');
    $('doctor-meta-wrap').style.display = 'none';
  }
}

document.addEventListener('DOMContentLoaded', async () => {
  renderNavbar('schedules');
  await loadDoctors();

  $('doctor-select').addEventListener('change', (e) => handleDoctorChange(e.target.value));

  $('btn-reset').addEventListener('click', () => {
    if (!currentDoctorId) return;
    editState = buildEditState(currentSchedule || []);
    $('form-alert').classList.add('d-none');
    renderAll();
  });

  $('btn-save').addEventListener('click', save);
  $('btn-copy-mon').addEventListener('click', applyMondayToAll);
  $('btn-enable-all').addEventListener('click', enableAll);
  $('btn-weekend-rest').addEventListener('click', weekendRest);

  window.addEventListener('beforeunload', (e) => {
    if (dirty()) {
      e.preventDefault();
      e.returnValue = '';
    }
  });
});
