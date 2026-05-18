package com.medicita.app.dto.schedule;

import lombok.*;

import java.time.LocalTime;
import java.util.List;

@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class DoctorAvailabilityDTO {

    private String weekDay;
    private boolean working;
    private boolean onLeave;
    private LocalTime startTime;
    private LocalTime endTime;
    private List<Slot> slots;

    @Getter
    @Setter
    @NoArgsConstructor
    @AllArgsConstructor
    @Builder
    public static class Slot {
        private String time;     // "HH:mm"
        private boolean booked;
    }
}
