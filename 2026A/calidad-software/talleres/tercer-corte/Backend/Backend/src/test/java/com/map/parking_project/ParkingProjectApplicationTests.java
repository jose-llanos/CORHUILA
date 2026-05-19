package com.map.parking_project;

import org.junit.jupiter.api.Test;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.test.mock.mockito.MockBean;
import org.springframework.mail.javamail.JavaMailSender;
import javax.sql.DataSource;

@SpringBootTest(properties = {
    "spring.jpa.hibernate.ddl-auto=none"
})
class ParkingProjectApplicationTests {

    @MockBean
    DataSource dataSource;

    @MockBean
    JavaMailSender mailSender;

    @Test
    void contextLoads() {
    }
}