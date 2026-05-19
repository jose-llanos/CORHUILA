package com.corhuila.calidad;

import java.util.Objects;
import java.util.logging.Logger;

public class BankAccount {
    private static final Logger LOGGER = Logger.getLogger(BankAccount.class.getName());

    private double balance;
    private final String accountNumber;

    public BankAccount(String accountNumber, double initialBalance) {
        this.accountNumber = Objects.requireNonNull(accountNumber, "El número de cuenta no puede ser nulo");
        this.balance = initialBalance;
    }

    public void deposit(double amount) {
        if (amount <= 0) {
            LOGGER.warning("La cantidad a depositar debe ser mayor que cero.");
            return;
        }

        balance += amount;
    }

    public boolean isSameAccount(BankAccount other) {
        if (other == null) {
            return false;
        }
        return this.accountNumber.equals(other.accountNumber);
    }

    public double getBalance() {
        return balance;
    }

    public String getAccountNumber() {
        return accountNumber;
    }
}