package com.autospark.migueljuliana;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.data.jpa.repository.config.EnableJpaAuditing;

@SpringBootApplication
@EnableJpaAuditing(auditorAwareRef = "auditorAwareImpl")
public class MigueljulianaApplication {

	public static void main(String[] args) {
		SpringApplication.run(MigueljulianaApplication.class, args);
	}
	// Método principal vacío intencionalmente.
	// Esta clase se utiliza únicamente como punto de entrada del proyecto.
}
