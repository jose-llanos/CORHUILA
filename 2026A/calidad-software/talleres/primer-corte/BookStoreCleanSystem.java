/**
 * SISTEMA DE LIBRERÍA - EJEMPLO DE CÓDIGO LIMPIO
 */
public class BookStoreCleanSystem {

    public static void main(String[] args) {

        // Inyección de dependencias
        InventoryService inventory = new InventoryService();
        NotificationService notifier = new NotificationService();
        OrderProcessor processor = new OrderProcessor(inventory, notifier);

        System.out.println("=== INICIO DE PRUEBAS DEL SISTEMA ===\n");

        // CASO 1: Pedido correcto
        try {

            Customer customer = new Customer(
                    "Juan Perez",
                    "juan@corhuila.edu.co"
            );

            Book book = new Book(
                    "Clean Code",
                    50.00,
                    "REF-9988"
            );

            Order order = new Order(customer, book, 2);

            processor.processOrder(order);

            System.out.println("Pedido procesado correctamente");

        } catch (OrderException e) {

            System.err.println("ERROR: " + e.getMessage());

        }

        // CASO 2: Libro sin stock
        try {

            Customer customer2 = new Customer(
                    "Maria Gomez",
                    "maria@mail.com"
            );

            Book bookNoStock = new Book(
                    "Libro Agotado",
                    20.00,
                    "REF-0000"
            );

            Order order2 = new Order(customer2, bookNoStock, 1);

            processor.processOrder(order2);

        } catch (OrderException e) {

            System.out.println("Sistema detectó falta de stock");
            System.out.println("Mensaje: " + e.getMessage());

        }
    }

    // -----------------------
    // CLASE CUSTOMER
    // -----------------------

    static class Customer {

        private final String name;
        private final String email;

        public Customer(String name, String email) {
            this.name = name;
            this.email = email;
        }

        public String getEmail() {
            return email;
        }
    }

    // -----------------------
    // CLASE BOOK
    // -----------------------

    static class Book {

        private final String title;
        private final double basePrice;
        private final String sku;

        public Book(String title, double basePrice, String sku) {
            this.title = title;
            this.basePrice = basePrice;
            this.sku = sku;
        }

        public double getBasePrice() {
            return basePrice;
        }

        public String getSku() {
            return sku;
        }
    }

    // -----------------------
    // CLASE ORDER
    // -----------------------

    static class Order {

        private final Customer customer;
        private final Book book;
        private final int quantity;

        public Order(Customer customer, Book book, int quantity) {
            this.customer = customer;
            this.book = book;
            this.quantity = quantity;
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

    // -----------------------
    // EXCEPCIONES
    // -----------------------

    static class OrderException extends RuntimeException {

        public OrderException(String message) {
            super(message);
        }
    }

    static class OutOfStockException extends OrderException {

        public OutOfStockException(String sku) {
            super("Sin stock disponible para el libro con SKU: " + sku);
        }
    }

    // -----------------------
    // ORDER PROCESSOR
    // -----------------------

    static class OrderProcessor {

        private static final double TAX_RATE = 1.19;

        private final InventoryService inventoryService;
        private final NotificationService notificationService;

        public OrderProcessor(
                InventoryService inventoryService,
                NotificationService notificationService
        ) {
            this.inventoryService = inventoryService;
            this.notificationService = notificationService;
        }

        public void processOrder(Order order) {

            validateOrder(order);
            checkStock(order);

            double total = calculateTotal(order);

            completeTransaction(order, total);
        }

        private void validateOrder(Order order) {

            if (order == null) {
                throw new OrderException("La orden no puede ser nula");
            }

        }

        private void checkStock(Order order) {

            if (!inventoryService.hasStock(order.getBook().getSku())) {

                throw new OutOfStockException(order.getBook().getSku());

            }
        }

        private double calculateTotal(Order order) {

            double subtotal =
                    order.getBook().getBasePrice()
                            * order.getQuantity();

            return subtotal * TAX_RATE;
        }

        private void completeTransaction(Order order, double total) {

            notificationService.sendConfirmation(
                    order.getCustomer(),
                    total
            );

        }
    }

    // -----------------------
    // INVENTORY SERVICE
    // -----------------------

    static class InventoryService {

        public boolean hasStock(String sku) {

            return !"REF-0000".equals(sku);

        }
    }

    // -----------------------
    // NOTIFICATION SERVICE
    // -----------------------

    static class NotificationService {

        public void sendConfirmation(Customer customer, double total) {

            System.out.println("Enviando correo a: " + customer.getEmail());
            System.out.println("Total de la compra: $" + total);

        }
    }
}