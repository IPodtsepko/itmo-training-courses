package expression.generic;

import expression.CommonExpression;
import expression.parser.ExpressionParser;
import expression.parser.exceptions.ParseException;
import expression.solvers.*;
import expression.solvers.exceptions.CalculationException;

import java.util.Map;
import java.util.Objects;

/**
 * Title task: "Homework 5. Computing in various types: generics"
 * @author Igor Podtsepko (i.podtsepko@outlook.com)
 */

public class GenericTabulator implements Tabulator {
    private final static Map<String, Solver<?>> SOLVERS = Map.of(
            "i", ISolver.INSTANCE,
            "d", DSolver.INSTANCE,
            "bi", BISolver.INSTANCE,
            "l", LSolver.INSTANCE,
            "s", SSolver.INSTANCE,
            "u", USolver.INSTANCE
    );

    public static void main(final String[] args) {
        if (args.length != 2 || Objects.isNull(args[0]) || Objects.isNull(args[1])) {
            System.out.println("Enter 2 arguments");
            return;
        }
        final String mode = args[0].substring(1);
        if (!SOLVERS.containsKey(mode)) {
            System.out.println("Unsupported mode " + mode);
            return;
        }
        final Object[][][] table = new GenericTabulator().tabulate(mode, args[1], -2, 2, -2, 2, -2, 2);
        for (int x = -2; x < 3; x++) {
            for (int y = -2; y < 3; y++) {
                for (int z = -2; z < 3; z++) {
                    System.out.println("(" + x + ", " + y + ", " + z + "): " + table[x + 2][y + 2][z + 2]);
                }
            }
        }
    }

    @Override
    public Object[][][] tabulate(
            final String mode,
            final String expression,
            final int x1, final int x2, final int y1, final int y2, final int z1, final int z2) throws ParseException {
        return tabulate(new ExpressionParser().parse(expression), x1, x2, y1, y2, z1, z2, SOLVERS.get(mode));
    }

    private static <T> Object[][][] tabulate(
            final CommonExpression expr,
            final int x1, final int x2, final int y1, final int y2, final int z1, final int z2,
            final Solver<T> solver
    ) {
        final int x = (x2 + 1) - x1;
        final int y = (y2 + 1) - y1;
        final int z = (z2 + 1) - z1;

        final Object[][][] table = new Object[x][y][z];

        for (int dx = 0; dx < x; dx++) {
            for (int dy = 0; dy < y; dy++) {
                for (int dz = 0; dz < z; dz++) {
                    try {
                        table[dx][dy][dz] = expr.evaluate(
                                solver.valueOf(x1 + dx),
                                solver.valueOf(y1 + dy),
                                solver.valueOf(z1 + dz),
                                solver
                        );
                    } catch (final CalculationException ignored) {
                    }
                }
            }
        }

        return table;
    }
}
