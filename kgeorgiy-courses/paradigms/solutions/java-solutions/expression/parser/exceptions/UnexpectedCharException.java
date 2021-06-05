package expression.parser.exceptions;

/**
 * Title task: "Homework 5. Computing in various types: generics"
 * @author Igor Podtsepko (i.podtsepko@outlook.com)
 */

public class UnexpectedCharException extends ParseException {
    public UnexpectedCharException(char excepted, char encountered, int position) {
        super(String.format("expected '%c', encountered '%c'", excepted, encountered), position);
    }
}
