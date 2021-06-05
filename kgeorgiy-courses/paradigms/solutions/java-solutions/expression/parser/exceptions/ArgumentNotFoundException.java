package expression.parser.exceptions;

/**
 * Title task: "Homework 5. Computing in various types: generics"
 * @author Igor Podtsepko (i.podtsepko@outlook.com)
 */

public class ArgumentNotFoundException extends ParseException {
    public ArgumentNotFoundException(int position) {
        super("argument expected", position);
    }
}
