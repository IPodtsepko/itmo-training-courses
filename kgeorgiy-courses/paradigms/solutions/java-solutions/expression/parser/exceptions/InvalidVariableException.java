package expression.parser.exceptions;

/**
 * Title task: "Homework 5. Computing in various types: generics"
 * @author Igor Podtsepko (i.podtsepko@outlook.com)
 */

public class InvalidVariableException extends ParseException {
    public InvalidVariableException(String parsed, int position) {
        super(String.format("invalid variable encountered - %s", parsed), position);
    }
}
