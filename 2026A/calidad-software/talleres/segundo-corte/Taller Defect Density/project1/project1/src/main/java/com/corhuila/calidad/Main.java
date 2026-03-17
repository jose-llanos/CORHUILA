package com.corhuila.calidad;

import java.util.logging.Logger;
import java.util.logging.Level;

public class Main {

    private static final Logger logger = Logger.getLogger(Main.class.getName());

    public static void main(String[] args) {
        logger.info("Hello and welcome!");

        for (int i = 1; i <= 5; i++) {
            logger.log(Level.INFO, "i = {0}", i);
        }
    }
}