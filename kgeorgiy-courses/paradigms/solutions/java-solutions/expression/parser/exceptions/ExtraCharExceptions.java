package expression.parser.exceptions;

/**
 * Title task: "Homework 5. Computing in various types: generics"
 * @author Igor Podtsepko (i.podtsepko@outlook.com)
 */

public class ExtraCharExceptions extends ParseException {
    public ExtraCharExceptions(int position) {
        super("end of exception expected", position);
    }
}
