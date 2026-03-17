package com.corhuila.calidad;
import java.util.logging.Logger;
public class BankAccount {
    private double balance;
    private String accountNumber;

    private double balancePublic = 0.0;
    private static final Logger logger = Logger.getLogger(BankAccount.class.getName());

    public BankAccount(String accountNumber, double initialBalance) {
        this.accountNumber = accountNumber;
        this.balance = initialBalance;
    }

    public double getBalancePublic() {
        return balancePublic;
    }

    public void setBalancePublic(double balancePublic) {
        this.balancePublic = balancePublic;
    }

    public void deposit(double amount) {
        if (amount < 0) {
            logger.warning("Error: cantidad negativa");
            return;
        }
        balance = balance + amount;
    }

    public boolean isSameAccount(BankAccount other) {
        return this.accountNumber != null
                && this.accountNumber.equals(other.accountNumber);
    }

    public double getBalance() {
        return balance;
    }
}