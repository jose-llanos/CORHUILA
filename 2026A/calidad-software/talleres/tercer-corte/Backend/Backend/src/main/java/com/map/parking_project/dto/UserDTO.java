package com.map.parking_project.dto;

// Le dice a SonarQube que ignore todos los code smells de este archivo, pero sí medirá su cobertura.
@SuppressWarnings("all")
public class UserDTO {
    private String name;
    private String lastname;
    private String phone;
    private String plate;
    private String typecar;
    private String email;
    private String password;
    private String rol;

    // Genera los Getters y Setters para todos los campos
    public String getName() { return name; }
    public void setName(String name) { this.name = name; }

    public String getLastname() { return lastname; }
    public void setLastname(String lastname) { this.lastname = lastname; }

    public String getPhone() { return phone; }
    public void setPhone(String phone) { this.phone = phone; }

    public String getPlate() { return plate; }
    public void setPlate(String plate) { this.plate = plate; }

    public String getTypecar() { return typecar; }
    public void setTypecar(String typecar) { this.typecar = typecar; }

    public String getEmail() { return email; }
    public void setEmail(String email) { this.email = email; }

    public String getPassword() { return password; }
    public void setPassword(String password) { this.password = password; }

    public String getRol() { return rol; }
    public void setRol(String rol) { this.rol = rol; }
}