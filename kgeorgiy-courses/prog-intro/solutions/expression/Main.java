package expression;

import expression.exceptions.CheckedAdd;
import expression.exceptions.CheckedBinaryOperation;
import expression.exceptions.CheckedMultiply;
import expression.exceptions.CheckedSubtract;

/**
 * @author Igor Podtsepko (i.podtsepko@niuitmo.com)
 */
public class Main {
    public static void main(String[] args) {
        Variable x = new Variable("x");
        CheckedBinaryOperation test = new CheckedAdd(
                new CheckedSubtract(
                        new CheckedMultiply(x, x),
                        new CheckedMultiply(new Const(2), x)),
                new Const(1)
        );
        System.out.println(test.evaluate(Integer.parseInt(args[0])));
    }
}
