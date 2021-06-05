package expression.exceptions.parse_exceptions;

/**
 * @author Igor Podtsepko (i.podtsepko@niuitmo.com)
 */
public class InvalidVariableException extends ParseException {
    public InvalidVariableException(String parsed, int position) {
        super(String.format("invalid variable encountered - %s", parsed), position);
    }
}
