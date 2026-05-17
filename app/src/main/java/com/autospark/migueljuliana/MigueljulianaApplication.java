package com.autospark.migueljuliana;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.context.annotation.Bean;
import org.springframework.data.domain.AuditorAware;
import org.springframework.data.jpa.repository.config.EnableJpaAuditing;

import java.util.Optional;

@SpringBootApplication
@EnableJpaAuditing(auditorAwareRef = "auditorAwareImpl")
public class MigueljulianaApplication {

	public static void main(String[] args) {
		SpringApplication.run(MigueljulianaApplication.class, args);
	}

	@Bean(name = "auditorAwareImpl")
	public AuditorAware<String> auditorAwareImpl() {
		return () -> Optional.of("system");
	}
}