package expression.parser.exceptions;

/**
 * Title task: "Homework 5. Computing in various types: generics"
 * @author Igor Podtsepko (i.podtsepko@outlook.com)
 */

abstract public class ParseException extends RuntimeException {
    public ParseException(final String message, int position) {
        super(String.format("%s (position = %s)", message, position));
    }
}
