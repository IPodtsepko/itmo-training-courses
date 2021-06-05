package expression.parser.exceptions;

/**
 * Title task: "Homework 5. Computing in various types: generics"
 * @author Igor Podtsepko (i.podtsepko@outlook.com)
 */

public class SpaceInNumberException extends ParseException {
    public SpaceInNumberException(int position) {
        super("space in number encountered", position);
    }
}
