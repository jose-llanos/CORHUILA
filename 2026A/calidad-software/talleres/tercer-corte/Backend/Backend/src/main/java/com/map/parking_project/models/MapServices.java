package com.map.parking_project.models;

import jakarta.persistence.*;

@Table(name = "servicios")
@Entity
public class MapServices {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)

    @Column(nullable = false)
    private Long id;
    @Column(nullable = false, length = 50)
    private String name;

    @Column(nullable = false, length = 50)
    private String price;

    @Column(nullable = false)
    private String description;
    public Long getId() {
        return id;
    }

    public void setId(Long id) {
        this.id = id;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public String getPrice() {
        return price;
    }

    public void setPrice(String price) {
        this.price = price;
    }

    public String getDescription() {
        return description;
    }

    public void setDescription(String description) {
        this.description = description;
    }

}
