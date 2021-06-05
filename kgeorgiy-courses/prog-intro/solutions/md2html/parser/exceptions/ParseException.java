package md2html.parser.exceptions;

/**
 * @author Georgiy Korneev (kgeorgiy@kgeorgiy.info)
 */
public class ParseException extends RuntimeException {
    public ParseException(final String message) {
        super(message);
    }
}
