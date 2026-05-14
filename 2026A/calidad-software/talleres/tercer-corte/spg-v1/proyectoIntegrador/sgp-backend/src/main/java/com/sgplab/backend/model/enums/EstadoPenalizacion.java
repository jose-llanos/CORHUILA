package com.sgplab.backend.model.enums;

public enum EstadoPenalizacion {
    ACTIVA,     // La penalización está en curso
    CUMPLIDA,   // El tiempo de penalización ya pasó
    LEVANTADA   // Un administrador perdonó/retiró la penalización antes de tiempo
}