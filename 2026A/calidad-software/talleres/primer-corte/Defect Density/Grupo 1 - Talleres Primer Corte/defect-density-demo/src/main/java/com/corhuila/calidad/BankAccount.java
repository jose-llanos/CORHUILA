package com.corhuila.calidad;

import java.util.Objects;
import java.util.logging.Logger;

public class BankAccount {

    private static final Logger logger =
            Logger.getLogger(BankAccount.class.getName());

    private double balance;
    private String accountNumber;

    private double balancePublic = 0.0;

    public BankAccount(String accountNumber, double initialBalance) {
        this.accountNumber = accountNumber;
        this.balance = initialBalance;
    }

    public double getBalancePublic() {
        return balancePublic;
    }

    public void setBalancePublic(double balancePublic) {
        if (balancePublic >= 0) {
            this.balancePublic = balancePublic;
        }
    }

    public void deposit(double amount) {
        if (amount < 0) {
            logger.warning("Error: cantidad negativa");
            return;
        }
        balance = balance + amount;
    }

    public boolean isSameAccount(BankAccount other) {
        if (other == null) {
            return false;
        }
        return Objects.equals(this.accountNumber, other.accountNumber);
    }

    public double getBalance() {
        return balance;
    }
}