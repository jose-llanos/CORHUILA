package com.medicita.app.service.impl;

import com.medicita.app.dto.schedule.DoctorAvailabilityDTO;
import com.medicita.app.dto.schedule.DoctorScheduleDTO;
import com.medicita.app.dto.schedule.DoctorScheduleRequest;
import com.medicita.app.entity.Appointment;
import com.medicita.app.entity.Doctor;
import com.medicita.app.entity.DoctorSchedule;
import com.medicita.app.enums.AppointmentStatus;
import com.medicita.app.enums.LeaveStatus;
import com.medicita.app.enums.Weekday;
import com.medicita.app.exception.ResourceNotFoundException;
import com.medicita.app.repository.AppointmentRepository;
import com.medicita.app.repository.DoctorLeaveRepository;
import com.medicita.app.repository.DoctorRepository;
import com.medicita.app.repository.DoctorScheduleRepository;
import com.medicita.app.service.DoctorScheduleService;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.LocalTime;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.UUID;
import java.util.stream.Collectors;

@Service
@RequiredArgsConstructor
@Transactional
public class DoctorScheduleServiceImpl implements DoctorScheduleService {

    private final DoctorScheduleRepository doctorScheduleRepository;
    private final DoctorRepository doctorRepository;
    private final AppointmentRepository appointmentRepository;
    private final DoctorLeaveRepository doctorLeaveRepository;

    @Override
    @Transactional(readOnly = true)
    public List<DoctorScheduleDTO> findByDoctor(UUID doctorId) {
        Doctor doctor = doctorRepository.findById(doctorId)
                .orElseThrow(() -> new ResourceNotFoundException("Doctor", "id", doctorId));
        return doctorScheduleRepository.findByDoctor(doctor).stream()
                .map(this::toDTO)
                .collect(Collectors.toList());
    }

    @Override
    public DoctorScheduleDTO create(UUID doctorId, DoctorScheduleRequest request) {
        Doctor doctor = doctorRepository.findById(doctorId)
                .orElseThrow(() -> new ResourceNotFoundException("Doctor", "id", doctorId));
        DoctorSchedule schedule = DoctorSchedule.builder()
                .doctor(doctor)
                .weekDay(request.getWeekDay())
                .startTime(request.getStartTime())
                .endTime(request.getEndTime())
                .build();
        return toDTO(doctorScheduleRepository.save(schedule));
    }

    @Override
    public DoctorScheduleDTO update(UUID scheduleId, DoctorScheduleRequest request) {
        DoctorSchedule schedule = doctorScheduleRepository.findById(scheduleId)
                .orElseThrow(() -> new ResourceNotFoundException("DoctorSchedule", "id", scheduleId));
        schedule.setWeekDay(request.getWeekDay());
        schedule.setStartTime(request.getStartTime());
        schedule.setEndTime(request.getEndTime());
        return toDTO(doctorScheduleRepository.save(schedule));
    }

    @Override
    public void delete(UUID scheduleId) {
        DoctorSchedule schedule = doctorScheduleRepository.findById(scheduleId)
                .orElseThrow(() -> new ResourceNotFoundException("DoctorSchedule", "id", scheduleId));
        schedule.setActive(false);
        doctorScheduleRepository.save(schedule);
    }

    @Override
    public List<DoctorScheduleDTO> replaceWeekly(UUID doctorId, List<DoctorScheduleRequest> weekly) {
        Doctor doctor = doctorRepository.findById(doctorId)
                .orElseThrow(() -> new ResourceNotFoundException("Doctor", "id", doctorId));

        List<DoctorSchedule> existing = doctorScheduleRepository.findByDoctor(doctor);

        for (DoctorScheduleRequest req : weekly) {
            if (req.getStartTime() != null && req.getEndTime() != null
                    && !req.getStartTime().isBefore(req.getEndTime())) {
                throw new IllegalArgumentException(
                        "startTime must be before endTime for " + req.getWeekDay());
            }

            DoctorSchedule entry = existing.stream()
                    .filter(s -> s.getWeekDay() == req.getWeekDay())
                    .findFirst()
                    .orElse(null);

            if (entry == null) {
                entry = DoctorSchedule.builder()
                        .doctor(doctor)
                        .weekDay(req.getWeekDay())
                        .startTime(req.getStartTime())
                        .endTime(req.getEndTime())
                        .active(req.getActive() == null || req.getActive())
                        .build();
            } else {
                entry.setStartTime(req.getStartTime());
                entry.setEndTime(req.getEndTime());
                entry.setActive(req.getActive() == null || req.getActive());
            }
            doctorScheduleRepository.save(entry);
        }

        return doctorScheduleRepository.findByDoctor(doctor).stream()
                .map(this::toDTO)
                .collect(Collectors.toList());
    }

    @Override
    @Transactional(readOnly = true)
    public DoctorAvailabilityDTO getAvailability(UUID doctorId, LocalDate date) {
        Doctor doctor = doctorRepository.findById(doctorId)
                .orElseThrow(() -> new ResourceNotFoundException("Doctor", "id", doctorId));

        Weekday weekday = Weekday.valueOf(date.getDayOfWeek().name());

        DoctorSchedule schedule = doctorScheduleRepository
                .findByDoctorAndWeekDay(doctor, weekday)
                .orElse(null);

        boolean onLeave = doctorLeaveRepository.findByDoctorAndStatus(doctor, LeaveStatus.APPROVED)
                .stream()
                .anyMatch(l -> !date.isBefore(l.getStartDate()) && !date.isAfter(l.getEndDate()));

        DoctorAvailabilityDTO.DoctorAvailabilityDTOBuilder dto = DoctorAvailabilityDTO.builder()
                .weekDay(weekday.name())
                .onLeave(onLeave);

        if (schedule == null || !schedule.isActive()) {
            return dto.working(false).slots(List.of()).build();
        }

        dto.working(true)
                .startTime(schedule.getStartTime())
                .endTime(schedule.getEndTime());

        Set<LocalTime> booked = new HashSet<>();
        LocalDateTime dayStart = date.atStartOfDay();
        LocalDateTime dayEnd = date.plusDays(1).atStartOfDay();
        List<Appointment> dayAppts = appointmentRepository
                .findByDoctorAndDateTimeBetween(doctor, dayStart, dayEnd);
        for (Appointment a : dayAppts) {
            if (a.getStatus() == AppointmentStatus.PENDING
                    || a.getStatus() == AppointmentStatus.CONFIRMED) {
                booked.add(a.getDateTime().toLocalTime().withMinute(0).withSecond(0).withNano(0));
            }
        }

        List<DoctorAvailabilityDTO.Slot> slots = new ArrayList<>();
        LocalTime t = schedule.getStartTime().withMinute(0).withSecond(0).withNano(0);
        while (t.isBefore(schedule.getEndTime())) {
            slots.add(DoctorAvailabilityDTO.Slot.builder()
                    .time(String.format("%02d:%02d", t.getHour(), t.getMinute()))
                    .booked(onLeave || booked.contains(t))
                    .build());
            t = t.plusHours(1);
        }

        return dto.slots(slots).build();
    }

    private DoctorScheduleDTO toDTO(DoctorSchedule schedule) {
        return DoctorScheduleDTO.builder()
                .id(schedule.getId())
                .dayOfWeek(schedule.getWeekDay().name())
                .startTime(schedule.getStartTime())
                .endTime(schedule.getEndTime())
                .active(schedule.isActive())
                .build();
    }
}
