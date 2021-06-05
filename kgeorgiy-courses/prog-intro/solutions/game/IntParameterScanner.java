package game;

import java.io.PrintStream;

public class IntParameterScanner {

    private final Scanner sc;
    private final PrintStream out;

    public IntParameterScanner(final Scanner sc, final PrintStream out) {
        this.sc = sc;
        this.out = out;
    }

    public interface Checker {
        boolean isSuitable(int x);
    }

    public int scanParameter(final String call) {
        return scanParameter(
                call, "Is invalid value", x -> true);
    }

    public int scanParameter(final String call, final String errorMessage, Checker checker) {
        while (true) {
            int scanned = scanInt(call);
            if (checker.isSuitable(scanned)) {
                return scanned;
            }
            out.println(errorMessage);
        }
    }

    private int scanInt(final String call) {
        while (true) {
            out.print(call);
            if (sc.hasNextIntInLine()) {
                int result = sc.nextIntInLine();
                if (!sc.hasNextInLine()) {
                    sc.goToNextLine();
                    return result;
                }
            }
            if (!sc.hasNextLine()) throw new AssertionError("Stream closed");
            out.println("Enter one integer");
            sc.goToNextLine();
        }
    }
}
