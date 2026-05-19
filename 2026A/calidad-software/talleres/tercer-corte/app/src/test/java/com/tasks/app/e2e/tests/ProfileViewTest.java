package com.tasks.app.e2e.tests;
 
import com.tasks.app.e2e.config.BaseE2ETest;
import com.tasks.app.e2e.config.BrowserType;
import com.tasks.app.e2e.pages.DashboardPage;
import com.tasks.app.e2e.pages.LoginPage;
import com.tasks.app.e2e.pages.ProfileModalPage;
import com.tasks.app.e2e.utils.TestDataFactory;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;
 
import static org.assertj.core.api.Assertions.assertThat;
 
/**
 * Caso 5 (TC-PROF-001) — RF-01.4.
 */
class ProfileViewTest extends BaseE2ETest {
 
    @ParameterizedTest(name = "Visualizar perfil del usuario en {0}")
    @EnumSource(BrowserType.class)
    void shouldDisplayUserProfileModal(BrowserType browser) {
        setupDriver(browser);
 
        String user = TestDataFactory.uniqueUsername();
        String email = TestDataFactory.emailFor(user);
        String pass = TestDataFactory.DEFAULT_PASSWORD;
 
        new LoginPage(driver, wait, baseUrl).open()
                .goToRegister()
                .register(user, email, pass);
 
        DashboardPage dashboard = new LoginPage(driver, wait, baseUrl).open()
                .login(user, pass);
 
        ProfileModalPage profile = dashboard.openProfileModal();
 
        assertThat(profile.getUsername()).contains(user);
        assertThat(profile.getEmail()).contains(email);
        assertThat(profile.getId()).isNotBlank();
        assertThat(profile.getCreatedAt()).isNotBlank();
    }
}