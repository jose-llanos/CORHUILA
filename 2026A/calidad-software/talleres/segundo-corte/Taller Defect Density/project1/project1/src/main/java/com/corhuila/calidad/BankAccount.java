package com.corhuila.calidad;

import java.util.logging.Logger;

public class BankAccount {

    private static final Logger logger = Logger.getLogger(BankAccount.class.getName());

    private double balance;
    private String accountNumber;

    // Constructor
    public BankAccount(String accountNumber, double initialBalance) {
        this.accountNumber = accountNumber;
        this.balance = initialBalance;
    }

    // DEFECTO 8 corregido: lógica y logging
    public void deposit(double amount) {
        if (amount < 0) {
            logger.severe("Error: cantidad negativa");
            return;
        }
        balance += amount;
    }

    // DEFECTO 9 corregido: uso de equals()
    public boolean isSameAccount(BankAccount other) {
        return this.accountNumber.equals(other.accountNumber);
    }

    // DEFECTO 10 corregido: código limpio
    public double getBalance() {
        return balance;
    }
}