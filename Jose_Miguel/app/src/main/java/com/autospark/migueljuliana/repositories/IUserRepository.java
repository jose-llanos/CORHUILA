package com.autospark.migueljuliana.repositories;

import java.util.Optional;

import org.springframework.data.repository.CrudRepository;

import com.autospark.migueljuliana.models.User;

public interface IUserRepository extends CrudRepository<User, Long> {

    boolean existsByEmail(String email);

    Optional<User> findByEmail(String email);

    Optional<User> findByLicensePlate(String licensePlate);
}