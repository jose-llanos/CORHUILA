# Pruebas funcionales — Selenium WebDriver (POM)

Automatización UI con **Brave Browser** (Chromium + ChromeDriver).

## Requisitos previos

1. **Brave** instalado: https://brave.com/download/
2. **Backend** (Docker): `cd app/Docker && docker compose up -d`
3. **Frontend**: `cd app/Front/frontend && npm start` → http://localhost:4200

## Ejecución

```powershell
cd app/tests/selenium
.\run-selenium.ps1
```

O manualmente:

```bash
mvn test -Dbrowser=brave
```

Modo headless: `mvn test -Dbrowser=brave -Dheadless=true`

## Escenarios (7)

| # | Escenario |
|---|-----------|
| 01 | Home muestra bienvenida |
| 02 | Home → Login |
| 03 | Login inválido → error |
| 04 | Login admin → `/admin` |
| 05 | Home → Servicios |
| 06 | Login → Recuperar contraseña |
| 07 | Login → Registro |

## Usuario de prueba (API)

| Email | Contraseña | Rol |
|-------|------------|-----|
| `selenium.admin@test.com` | `Test1234!` | Administrador |

## Reportes

`app/reports/selenium/surefire-reports/`

## CI (GitHub Actions)

En la nube se usa Chrome headless; en tu PC solo **Brave**.
