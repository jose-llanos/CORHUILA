package com.autospark.migueljuliana.models;

import jakarta.persistence.*;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.hibernate.envers.Audited;

import java.io.Serializable;

@Data
@NoArgsConstructor
@AllArgsConstructor
@Entity
@Audited
@Table(name = "services")
public class CarWashService implements Serializable {

    private static final long serialVersionUID = 1L;

    @Id
    @Column(nullable = false, name = "Id_Servicios")
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(name = "name_service", length = 200)
    private String name;

    @Column(name = "descripcion", length = 200)
    private String description;

    @Column(name = "precio_service", nullable = false)
    private Double price;

    @Column(name = "estado_service", nullable = false)
    private boolean active;

    @Column(name = "url_img")
    private String imageUrl;
}