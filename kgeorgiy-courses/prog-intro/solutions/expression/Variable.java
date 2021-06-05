package expression;

/**
 * @author Igor Podtsepko (i.podtsepko@niuitmo.com)
 */
public class Variable extends Value {
    private final String name;

    public Variable(String name) {
        this.name = name;
    }

    @Override
    public int evaluate(int value) {
        return value;
    }

    @Override
    public int evaluate(int x, int y, int z) {
        switch (name) {
            case "x":
                return x;
            case "y":
                return y;
            case "z":
                return z;
            default:
                throw new AssertionError(String.format("Illegal name of variable: \"%s\"", name));
        }
    }

    @Override
    public String toString() {
        return name;
    }

    @Override
    public boolean equals(Object other) {
        if (this == other) {
            return true;
        }
        if (other == null || getClass() != other.getClass()) {
            return false;
        }
        Variable that = (Variable) other;
        return name.equals(that.name);
    }

    @Override
    public int hashCode() {
        return name.hashCode();
    }

    @Override
    public double evaluate(double value) {
        return value;
    }

    @Override
    public PrioritiesPattern getPriority() {
        return PrioritiesPattern.VALUE;
    }
}
