package expression.exceptions.parse_exceptions;

/**
 * @author Igor Podtsepko (i.podtsepko@niuitmo.com)
 */
public class OpenParenthesisMissedException extends ParseException {
    public OpenParenthesisMissedException(int position) {
        super("opening parenthesis for current closing missed", position);
    }
}
