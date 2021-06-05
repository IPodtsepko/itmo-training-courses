package expression.exceptions.parse_exceptions;

/**
 * @author Igor Podtsepko (i.podtsepko@niuitmo.com)
 */
public class UnexpectedCharException extends ParseException {
    public UnexpectedCharException(char excepted, char encountered, int position) {
        super(String.format("expected '%c', encountered '%c'", excepted, encountered), position);
    }
}
