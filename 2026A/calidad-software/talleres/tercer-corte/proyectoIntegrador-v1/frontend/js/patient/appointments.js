import { get, post, put } from '../utils/api.js';
import { requireAuth } from '../utils/auth.js';
import { renderNavbar } from '../utils/navbar.js';
import { showToast, showConfirm } from '../utils/toast.js';

requireAuth('PATIENT');

function parseDateTime(dt) {
  if (Array.isArray(dt)) { const [y, mo, d, h = 0, mi = 0] = dt; return new Date(y, mo - 1, d, h, mi); }
  return new Date(dt);
}

function fmt(dt) {
  return parseDateTime(dt).toLocaleDateString('es-CO', {
    year: 'numeric', month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit',
  });
}

function statusBadge(status) {
  const map = { PENDING: 'status-pending', CONFIRMED: 'status-confirmed', COMPLETED: 'status-completed', CANCELLED: 'status-cancelled' };
  return `<span class="badge ${map[status] || 'bg-secondary'}">${status}</span>`;
}

// ── Active appointments table ─────────────────────────────────────────────

let currentPage = 0;
let totalPages  = 1;

async function loadAppointments(page = 0) {
  document.getElementById('table-spinner').classList.remove('d-none');
  document.getElementById('table-wrapper').classList.add('d-none');

  const data = await get(`/patient/appointments?page=${page}&size=10`).catch(() => null);

  document.getElementById('table-spinner').classList.add('d-none');
  document.getElementById('table-wrapper').classList.remove('d-none');

  const content    = data?.content ?? [];
  totalPages       = data?.totalPages ?? 1;
  currentPage      = data?.page ?? page;
  const total      = data?.totalElements ?? content.length;

  document.getElementById('page-info').textContent =
    `Mostrando página ${currentPage + 1} de ${totalPages} (${total} cita${total !== 1 ? 's' : ''})`;
  document.getElementById('btn-prev-page').disabled = currentPage === 0;
  document.getElementById('btn-next-page').disabled = currentPage >= totalPages - 1;

  const active = content.filter(a => a.status === 'PENDING' || a.status === 'CONFIRMED');
  const tbody  = document.getElementById('appts-tbody');

  if (!active.length) {
    tbody.innerHTML = '<tr><td colspan="6" class="text-center text-muted py-4">No tienes citas activas.</td></tr>';
    return;
  }

  tbody.innerHTML = active.map(a => `
    <tr>
      <td class="fw-semibold">${a.doctorFullName}</td>
      <td>${a.specialtyName}</td>
      <td>${fmt(a.appointmentDateTime)}</td>
      <td>${a.reason || '–'}</td>
      <td>${statusBadge(a.status)}</td>
      <td>
        <button class="btn btn-sm btn-outline-danger" onclick="cancelAppt('${a.id}')">
          <i class="bi bi-x-lg"></i>
        </button>
      </td>
    </tr>`).join('');
}

window.cancelAppt = async function (id) {
  const ok = await showConfirm({
    title: 'Cancelar cita',
    message: '¿Seguro que deseas cancelar esta cita? Esta acción no se puede deshacer.',
    confirmText: 'Sí, cancelar',
    cancelText: 'No, mantener',
    variant: 'danger',
  });
  if (!ok) return;
  try {
    await put(`/patient/appointments/${id}/cancel`);
    showToast('Cita cancelada correctamente.', 'warning');
    await loadAppointments(currentPage);
  } catch (err) {
    showToast(err.message || 'Error al cancelar la cita.');
  }
};

// ── Wizard ────────────────────────────────────────────────────────────────

let selectedSpecialtyId = null;
let selectedDoctorId    = null;
let selectedDate        = null;
let selectedSlot        = null;
let currentAvailability = null;

const WEEKDAY_LABELS = {
  MONDAY: 'Lunes', TUESDAY: 'Martes', WEDNESDAY: 'Miércoles',
  THURSDAY: 'Jueves', FRIDAY: 'Viernes', SATURDAY: 'Sábado', SUNDAY: 'Domingo',
};

function setStep(n) {
  [1, 2, 3].forEach(i => {
    document.getElementById(`step-${i}`).classList.toggle('d-none', i !== n);
    const pill = document.getElementById(`step-${i}-pill`);
    pill.classList.toggle('active', i === n);
    pill.classList.toggle('done',   i < n);
  });
}

async function loadSpecialties() {
  const specialties = await get('/public/specialties').catch(() => []);
  const sel = document.getElementById('sel-specialty');
  sel.innerHTML = '<option value="">Selecciona una especialidad</option>' +
    (specialties || []).map(s => `<option value="${s.id}">${s.name}</option>`).join('');
}

function renderSlotsArea(html) {
  document.getElementById('slots-area').innerHTML = html;
}

function setBookEnabled(on) {
  document.getElementById('btn-book').disabled = !on;
}

async function loadAvailability(doctorId, date) {
  selectedSlot = null;
  setBookEnabled(false);
  renderSlotsArea('<div class="slot-info"><span class="spinner-border spinner-border-sm me-2"></span>Consultando disponibilidad…</div>');

  const data = await get(`/public/doctors/${doctorId}/availability?date=${date}`).catch(() => null);
  currentAvailability = data;

  if (!data) {
    renderSlotsArea('<div class="slot-empty">No se pudo obtener la disponibilidad. Intenta otra fecha.</div>');
    return;
  }

  const dayLabel = WEEKDAY_LABELS[data.weekDay] || data.weekDay;
  document.getElementById('appt-date-hint').textContent = `${dayLabel} · ${data.working ? 'Día laboral' : 'Día de descanso'}`;

  if (!data.working) {
    renderSlotsArea(`<div class="slot-empty">
      <i class="bi bi-moon me-1"></i>
      El médico no atiende los <strong>${dayLabel.toLowerCase()}</strong>. Elige otra fecha.
    </div>`);
    return;
  }

  if (data.onLeave) {
    renderSlotsArea(`<div class="slot-empty">
      <i class="bi bi-airplane me-1"></i>
      El médico está en permiso esa fecha. Elige otra.
    </div>`);
    return;
  }

  if (!data.slots || !data.slots.length) {
    renderSlotsArea('<div class="slot-empty">No hay horarios configurados para esta fecha.</div>');
    return;
  }

  // Filter out past slots if date is today
  const today = new Date();
  const isToday = date === today.toISOString().slice(0, 10);
  const nowH = today.getHours();
  const nowM = today.getMinutes();

  const html = `<div class="slot-grid">${data.slots.map(s => {
    const [h, m] = s.time.split(':').map(Number);
    const isPast = isToday && (h < nowH || (h === nowH && m <= nowM));
    const blocked = s.booked || isPast;
    return `<button type="button"
              class="time-slot ${blocked ? 'booked' : ''}"
              data-time="${s.time}"
              ${blocked ? 'disabled' : ''}
              title="${blocked ? (isPast ? 'Hora pasada' : 'Ocupado') : 'Disponible'}">
              ${s.time}
            </button>`;
  }).join('')}</div>`;
  renderSlotsArea(html);

  document.querySelectorAll('.time-slot:not(:disabled)').forEach(el => {
    el.addEventListener('click', () => {
      document.querySelectorAll('.time-slot.selected').forEach(s => s.classList.remove('selected'));
      el.classList.add('selected');
      selectedSlot = el.dataset.time;
      setBookEnabled(true);
    });
  });
}

async function loadDoctors(specialtyId) {
  const spinner = document.getElementById('doctors-spinner');
  const sel     = document.getElementById('sel-doctor');
  spinner.classList.remove('d-none');
  sel.classList.add('d-none');
  document.getElementById('btn-step2-next').disabled = true;

  const doctors = await get(`/public/specialties/${specialtyId}/doctors`).catch(() => []);

  spinner.classList.add('d-none');
  sel.classList.remove('d-none');
  sel.innerHTML = '<option value="">Selecciona un médico</option>' +
    (doctors || []).map(d => `<option value="${d.id}">${d.firstName} ${d.lastName}</option>`).join('');
}

document.addEventListener('DOMContentLoaded', async () => {
  renderNavbar('appointments');
  await loadAppointments(0);
  await loadSpecialties();

  document.getElementById('btn-prev-page').addEventListener('click', () => loadAppointments(currentPage - 1));
  document.getElementById('btn-next-page').addEventListener('click', () => loadAppointments(currentPage + 1));

  const today = new Date();
  const todayStr = today.toISOString().slice(0, 10);
  document.getElementById('appt-date').min = todayStr;

  document.getElementById('appt-date').addEventListener('change', async (e) => {
    selectedDate = e.target.value;
    if (selectedDate && selectedDoctorId) {
      await loadAvailability(selectedDoctorId, selectedDate);
    }
  });

  document.getElementById('sel-specialty').addEventListener('change', (e) => {
    document.getElementById('btn-step1-next').disabled = !e.target.value;
  });

  document.getElementById('btn-step1-next').addEventListener('click', async () => {
    selectedSpecialtyId = document.getElementById('sel-specialty').value;
    setStep(2);
    await loadDoctors(selectedSpecialtyId);
  });

  document.getElementById('sel-doctor').addEventListener('change', (e) => {
    document.getElementById('btn-step2-next').disabled = !e.target.value;
  });

  document.getElementById('btn-step2-back').addEventListener('click', () => setStep(1));

  document.getElementById('btn-step2-next').addEventListener('click', () => {
    selectedDoctorId = document.getElementById('sel-doctor').value;
    setStep(3);
    // Reset step-3 state
    selectedDate = null;
    selectedSlot = null;
    currentAvailability = null;
    document.getElementById('appt-date').value = '';
    document.getElementById('appt-date-hint').textContent = '';
    document.getElementById('appt-reason').value = '';
    setBookEnabled(false);
    renderSlotsArea('<div class="slot-info">Selecciona una fecha para ver los horarios disponibles.</div>');
  });

  document.getElementById('btn-step3-back').addEventListener('click', () => setStep(2));

  document.getElementById('btn-book').addEventListener('click', async () => {
    const date   = document.getElementById('appt-date').value;
    const reason = document.getElementById('appt-reason').value.trim();
    const btn    = document.getElementById('btn-book');

    if (!date || !selectedSlot) {
      showToast('Selecciona una fecha y un horario disponible.', 'warning');
      return;
    }
    if (!reason) {
      showToast('Indica el motivo de la consulta.', 'warning');
      return;
    }

    btn.disabled = true;
    btn.innerHTML = '<span class="spinner-border spinner-border-sm me-1"></span>Agendando...';

    try {
      const appointmentDateTime = `${date}T${selectedSlot}:00`;

      await post('/patient/appointments', {
        doctorId: selectedDoctorId,
        appointmentDateTime,
        reason,
      });

      showToast('¡Cita agendada exitosamente!', 'success');
      setStep(1);
      document.getElementById('sel-specialty').value = '';
      document.getElementById('appt-date').value     = '';
      document.getElementById('appt-reason').value   = '';
      document.getElementById('appt-date-hint').textContent = '';
      selectedSlot = null;
      document.getElementById('btn-step1-next').disabled = true;
      renderSlotsArea('<div class="slot-info">Selecciona una fecha para ver los horarios disponibles.</div>');
      await loadAppointments(0);
    } catch (err) {
      showToast(err.message || 'Error al agendar la cita.', 'danger');
    } finally {
      btn.innerHTML = '<i class="bi bi-calendar-check me-1"></i>Agendar cita';
      btn.disabled = !selectedSlot;
    }
  });
});
