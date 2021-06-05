package expression.interfaces;

import expression.CommonExpression;
import expression.parser.exceptions.ParseException;

/**
 * @author Georgiy Korneev (kgeorgiy@kgeorgiy.info)
 */

public interface Parser {
    CommonExpression parse(String expression) throws ParseException;
}