package md2html.parser.exceptions;

/**
 * @author Igor Podtsepko (i.podtsepko@niuitmo.com)
 */
public class UnexpectedCharException extends ParseException {
    public UnexpectedCharException(char expected, char actual, int position) {
        super("expected '" + expected + "', actual '" + actual + "' in position " + position);
    }
}
