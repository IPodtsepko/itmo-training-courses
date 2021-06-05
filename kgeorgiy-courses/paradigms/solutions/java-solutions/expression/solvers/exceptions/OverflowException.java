package expression.solvers.exceptions;

/**
 * Title task: "Homework 5. Computing in various types: generics"
 * @author Igor Podtsepko (i.podtsepko@outlook.com)
 */

public class OverflowException extends CalculationException {
    public OverflowException() {
        super("overflow");
    }
}