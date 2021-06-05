package expression.parser.exceptions;

/**
 * Title task: "Homework 5. Computing in various types: generics"
 * @author Igor Podtsepko (i.podtsepko@outlook.com)
 */

public class OpenParenthesisMissedException extends ParseException {
    public OpenParenthesisMissedException(int position) {
        super("opening parenthesis for current closing missed", position);
    }
}
