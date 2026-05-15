package com.sgplab.backend.model.enums;

public enum EstadoPrestamo {
    ACTIVO,     // El equipo está actualmente prestado
    DEVUELTO,   // El equipo ya fue regresado
    VENCIDO,    // Se pasó la fecha límite de entrega
    CANCELADO   // El préstamo no se concretó
}