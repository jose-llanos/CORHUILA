
import java.util.Optional;

/**
 * SISTEMA DE LIBRERÍA - EJEMPLO DE CÓDIGO LIMPIO
 */
public class BookStoreCleanSystem {

    public static void main(String[] args) {
        // 1. Configuración de dependencias (Inyección pura)
        InventoryService inventory = new InventoryService();
        NotificationService notifier = new NotificationService();
        OrderProcessor processor = new OrderProcessor(inventory, notifier);

        System.out.println("=== INICIO DE PRUEBAS DEL SISTEMA ===\n");

        // CASO DE PRUEBA 1: Flujo Exitoso
        try {
            Customer customer = new Customer("Juan Perez", "juan@corhuila.edu.co");
            Book book = new Book("Clean Code", 50.00, "REF-9988");
            Order order = new Order(customer, book, 2);

            processor.processOrder(order);
            System.out.println("✅ ÉXITO: Pedido procesado correctamente.");

        } catch (OrderException e) {
            System.err.println("❌ ERROR: " + e.getMessage());
        }

        // CASO DE PRUEBA 2: Fallo por Stock (Manejo de Errores)
        try {
            Customer c2 = new Customer("Maria G.", "maria@mail.com");
            Book noStockBook = new Book("Libro Agotado", 20.00, "REF-0000");
            Order order2 = new Order(c2, noStockBook, 1);

            processor.processOrder(order2); // Lanzará excepción

        } catch (OrderException e) {
            System.out.println("✅ CONTROLADO: El sistema detectó falta de stock.");
            System.out.println("   Mensaje: " + e.getMessage());
        }
    }

    // --- CAPA DE DOMINIO ---
    static class Customer {
        private final String name;
        private final String email;

        public Customer(String n, String e) {
            name = n;
            email = e;
        }

        public String getEmail() {
            return email;
        }
    }

    static class Book {
        private final String title;
        private final double basePrice;
        private final String sku;

        public Book(String t, double p, String s) {
            title = t;
            basePrice = p;
            sku = s;
        }

        public double getBasePrice() {
            return basePrice;
        }

        public String getSku() {
            return sku;
        }
    }

    static class Order {
        private final Customer customer;
        private final Book book;
        private final int quantity;

        public Order(Customer c, Book b, int q) {
            customer = c;
            book = b;
            quantity = q;
        }

        public Customer getCustomer() {
            return customer;
        }

        public Book getBook() {
            return book;
        }

        public int getQuantity() {
            return quantity;
        }
    }

    // --- EXCEPCIONES PERSONALIZADAS ---
    static class OrderException extends RuntimeException {
        public OrderException(String msg) {
            super(msg);
        }
    }

    static class OutOfStockException extends OrderException {
        public OutOfStockException(String sku) {
            super("Sin stock: " + sku);
        }
    }

    // --- SERVICIOS ---
    static class OrderProcessor {
        private static final double TAX_RATE = 1.19;
        private final InventoryService inventory;
        private final NotificationService notifier;

        public OrderProcessor(InventoryService inv, NotificationService notif) {
            this.inventory = inv;
            this.notifier = notif;
        }

        public void processOrder(Order order) {
            validateOrder(order);
            checkStock(order);
            double total = calculateTotal(order);
            completeTransaction(order, total);
        }

        private void validateOrder(Order order) {
            if (order == null)
                throw new OrderException("Orden nula");
        }

        private void checkStock(Order order) {
            if (!inventory.hasStock(order.getBook().getSku()))
                throw new OutOfStockException(order.getBook().getSku());
        }

        private double calculateTotal(Order order) {
            return order.getBook().getBasePrice() * order.getQuantity() * TAX_RATE;
        }

        private void completeTransaction(Order order, double total) {
            notifier.sendConfirmation(order.getCustomer(), total);
        }
    }

    static class InventoryService {
        public boolean hasStock(String sku) {
            return !"REF-0000".equals(sku);
        }
    }

    static class NotificationService {
        public void sendConfirmation(Customer c, double amount) {
            System.out.println("   📧 Enviando correo a: " + c.getEmail());
            System.out.println("   💰 Total confirmado: " + amount);
        }
    }
}
