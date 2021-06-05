package expression.solvers.exceptions;

/**
 * Title task: "Homework 5. Computing in various types: generics"
 * @author Igor Podtsepko (i.podtsepko@outlook.com)
 */

public class OutOfDefinitionException extends CalculationException {
    public OutOfDefinitionException(Object undefinedValue) {
        super("Out of scope of function definition for %d" + undefinedValue);
    }
}