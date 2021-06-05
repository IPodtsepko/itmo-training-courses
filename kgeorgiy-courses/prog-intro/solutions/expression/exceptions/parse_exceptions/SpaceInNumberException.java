package expression.exceptions.parse_exceptions;

/**
 * @author Igor Podtsepko (i.podtsepko@niuitmo.com)
 */
public class SpaceInNumberException extends ParseException {
    public SpaceInNumberException(int position) {
        super("space in number encountered", position);
    }
}
