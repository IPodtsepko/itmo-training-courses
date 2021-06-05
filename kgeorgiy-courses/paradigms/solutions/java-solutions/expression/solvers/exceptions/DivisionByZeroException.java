package expression.solvers.exceptions;

/**
 * Title task: "Homework 5. Computing in various types: generics"
 * @author Igor Podtsepko (i.podtsepko@outlook.com)
 */

public class DivisionByZeroException extends CalculationException {
    public DivisionByZeroException() {
        super("division by zero");
    }
}