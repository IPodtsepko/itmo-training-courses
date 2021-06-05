package expression.solvers.exceptions;

/**
 * Title task: "Homework 5. Computing in various types: generics"
 * @author Igor Podtsepko (i.podtsepko@outlook.com)
 */

abstract public class CalculationException extends RuntimeException {
    public CalculationException(String message) {
        super(message);
    }
}