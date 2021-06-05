package expression.exceptions.parse_exceptions;

/**
 * @author Georgiy Korneev (kgeorgiy@kgeorgiy.info)
 */
abstract public class ParseException extends RuntimeException {
    public ParseException(final String message, int position) {
        super(String.format("%s (position = %s)", message, position));
    }
}
