package com.corhuila.gestionpruebas.config;

import io.swagger.v3.oas.models.OpenAPI;
import io.swagger.v3.oas.models.info.Info;
import org.springdoc.core.models.GroupedOpenApi;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

@Configuration
public class OpenApiConfig {

    @Bean
    public OpenAPI customOpenAPI() {
        return new OpenAPI()
                .info(new Info()
                        .title("Clínica Veterinaria API")
                        .version("1.0")
                        .description("Endpoints del sistema de gestión veterinaria"));
    }

    @Bean
    public GroupedOpenApi publicApi() {
        return GroupedOpenApi.builder()
                .group("veterinaria")
                .pathsToMatch("/**")
                // ✅ Incluye controladores @Controller (Thymeleaf/MVC)
                .addOpenApiMethodFilter(method -> true)
                .build();
    }
}