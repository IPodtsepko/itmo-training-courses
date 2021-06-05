package expression.exceptions.parse_exceptions;

/**
 * @author Igor Podtsepko (i.podtsepko@niuitmo.com)
 */
public class ArgumentNotFoundException extends ParseException {
    public ArgumentNotFoundException(int position) {
        super("argument expected", position);
    }
}
