package expression.exceptions.parse_exceptions;

/**
 * @author Igor Podtsepko (i.podtsepko@niuitmo.com)
 */
public class ExtraCharExceptions extends ParseException {
    public ExtraCharExceptions(int position) {
        super("end of exception expected", position);
    }
}
