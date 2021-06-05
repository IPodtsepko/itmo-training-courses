package expression.solvers;

import expression.solvers.exceptions.DivisionByZeroException;
import expression.solvers.exceptions.OutOfDefinitionException;

import java.math.BigInteger;

/**
 * Title task: "Homework 5. Computing in various types: generics"
 * @author Igor Podtsepko (i.podtsepko@outlook.com)
 */

public class BISolver extends Solver<BigInteger> {
    public static Solver<BigInteger> INSTANCE = new BISolver();

    @Override
    public BigInteger add(BigInteger x, BigInteger y) {
        return x.add(y);
    }

    @Override
    public BigInteger subtract(BigInteger x, BigInteger y) {
        return x.subtract(y);
    }

    @Override
    public BigInteger divide(BigInteger x, BigInteger y) {
        if (y.equals(BigInteger.ZERO)) {
            throw new DivisionByZeroException();
        }
        return x.divide(y);
    }

    @Override
    public BigInteger multiple(BigInteger x, BigInteger y) {
        return x.multiply(y);
    }

    @Override
    public BigInteger mod(BigInteger x, BigInteger y) {
        if (y.equals(BigInteger.ZERO)) {
            throw new DivisionByZeroException();
        }
        if (y.compareTo(BigInteger.ZERO) < 0) {
            throw new OutOfDefinitionException(y);
        }
        return x.mod(y);
    }

    @Override
    public BigInteger negate(BigInteger x) {
        return x.negate();
    }

    @Override
    public BigInteger abs(BigInteger x) {
        return x.abs();
    }

    @Override
    public BigInteger square(BigInteger x) {
        return x.multiply(x);
    }

    @Override
    public BigInteger valueOf(int x) {
        return BigInteger.valueOf(x);
    }

    @Override
    public BigInteger parse(String value) {
        return new BigInteger(value);
    }
}
