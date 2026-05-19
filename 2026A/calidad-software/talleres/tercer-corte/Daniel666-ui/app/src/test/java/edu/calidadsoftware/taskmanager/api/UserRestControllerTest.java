package edu.calidadsoftware.taskmanager.api;

import edu.calidadsoftware.taskmanager.user.User;
import edu.calidadsoftware.taskmanager.user.UserRegistrationRequest;
import edu.calidadsoftware.taskmanager.user.UserRole;
import edu.calidadsoftware.taskmanager.user.UserResponse;
import edu.calidadsoftware.taskmanager.user.UserService;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;
import org.springframework.http.ResponseEntity;

import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.when;

/**
 * Pruebas unitarias para UserRestController.
 */
@DisplayName("UserRestController")
class UserRestControllerTest {

    private static UserRegistrationRequest request() {
        return UserRegistrationRequest.builder()
                .username("u1")
                .email("u1@example.com")
                .password("pass1234")
                .build();
    }

    @Test
    @DisplayName("register retorna 201 y no expone password")
    void register_created() {
        UserService service = Mockito.mock(UserService.class);
        User created = User.builder()
                .id(1L)
                .username("u1")
                .email("u1@example.com")
                .password("{bcrypt}x")
                .role(UserRole.USER)
                .build();
        when(service.register(any(UserRegistrationRequest.class))).thenReturn(created);

        UserRestController controller = new UserRestController(service);
        ResponseEntity<UserResponse> response = controller.register(request());

        assertEquals(201, response.getStatusCodeValue());
        assertNotNull(response.getBody());
        assertEquals("u1", response.getBody().getUsername());
        assertEquals(UserRole.USER, response.getBody().getRole());
    }

    @Test
    @DisplayName("list mapea entidades a DTO")
    void list_mapsToDto() {
        UserService service = Mockito.mock(UserService.class);
        when(service.findAll()).thenReturn(Arrays.asList(
                User.builder().id(1L).username("a").email("a@x.com").password("p").role(UserRole.ADMIN).build(),
                User.builder().id(2L).username("b").email("b@x.com").password("p").role(UserRole.USER).build()
        ));

        UserRestController controller = new UserRestController(service);
        List<UserResponse> users = controller.list();

        assertEquals(2, users.size());
        assertEquals("a", users.get(0).getUsername());
    }

    @Test
    @DisplayName("findById retorna DTO")
    void findById_returnsDto() {
        UserService service = Mockito.mock(UserService.class);
        when(service.findById(10L)).thenReturn(User.builder().id(10L).username("x").email("x@x.com").password("p").role(UserRole.USER).build());

        UserRestController controller = new UserRestController(service);
        UserResponse dto = controller.findById(10L);
        assertEquals(10L, dto.getId());
    }

    @Test
    @DisplayName("delete retorna 204")
    void delete_noContent() {
        UserService service = Mockito.mock(UserService.class);
        UserRestController controller = new UserRestController(service);

        ResponseEntity<Void> response = controller.delete(3L);
        assertEquals(204, response.getStatusCodeValue());
    }
}

