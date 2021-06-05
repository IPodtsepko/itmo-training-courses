package expression.parser.exceptions;

/**
 * Title task: "Homework 5. Computing in various types: generics"
 * @author Igor Podtsepko (i.podtsepko@outlook.com)
 */

public class InvalidConstException extends ParseException {
    public InvalidConstException(String parsed, int position) {
        super(String.format("invalid const encountered - %s", parsed), position);
    }
}
