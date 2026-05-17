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
@Table(name = "usuarios")
public class User implements Serializable {

    private static final long serialVersionUID = 1L;

    @Id
    @Column(nullable = false, name = "Id_Usuario")
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(name = "fullname", length = 100)
    private String fullName;

    @Column(name = "identity_card", length = 50)
    private String identityCard;

    @Column(length = 50, name = "email")
    private String email;

    @Column(name = "phone", length = 50)
    private String phone;

    @Column(name = "contrasenia")
    private String password;

    @Column(name = "licenseplate")
    private String licensePlate;

    @Enumerated(EnumType.STRING)
    @Column(nullable = false)
    private Role role;
}