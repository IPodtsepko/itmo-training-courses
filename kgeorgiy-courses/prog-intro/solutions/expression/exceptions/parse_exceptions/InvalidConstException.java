package expression.exceptions.parse_exceptions;

/**
 * @author Igor Podtsepko (i.podtsepko@niuitmo.com)
 */
public class InvalidConstException extends ParseException {
    public InvalidConstException(String parsed, int position) {
        super(String.format("invalid const encountered - %s", parsed), position);
    }
}
