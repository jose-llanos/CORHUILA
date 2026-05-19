package com.tasks.app.e2e.utils;
 
import java.util.concurrent.atomic.AtomicLong;
 
/**
 * Genera datos únicos para cada test. Como compartimos la BD del entorno
 * de desarrollo, necesitamos evitar colisiones por unicidad (username/email).
 */
public final class TestDataFactory {
 
    private static final AtomicLong COUNTER = new AtomicLong();
    public static final String DEFAULT_PASSWORD = "Pass1234";
 
    private TestDataFactory() {}
 
    public static String uniqueUsername() {
        return "user_" + System.currentTimeMillis() + "_" + COUNTER.incrementAndGet();
    }
 
    public static String emailFor(String username) {
        return username + "@test.local";
    }
 
    public static String uniqueProjectName() {
        return "Proyecto_" + System.currentTimeMillis() + "_" + COUNTER.incrementAndGet();
    }
 
    public static String uniqueTaskTitle() {
        return "Tarea_" + System.currentTimeMillis() + "_" + COUNTER.incrementAndGet();
    }
}
 