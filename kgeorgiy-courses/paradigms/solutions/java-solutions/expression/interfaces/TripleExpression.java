package expression.interfaces;

/**
 * @author Georgiy Korneev (kgeorgiy@kgeorgiy.info)
 */

public strictfp interface TripleExpression extends ToMiniString {
    int evaluate(int x, int y, int z);
}