package game;

import java.io.PrintStream;

public class HumanPlayer implements Player {
    private final PrintStream out;
    private final IntParameterScanner sc;

    public HumanPlayer(final PrintStream out, final Scanner in) {
        this.out = out;
        this.sc = new IntParameterScanner(in, out);
    }

    public HumanPlayer() {
        this(System.out, new Scanner(System.in));
    }

    @Override
    public Move move(final Position position, final Cell cell) {
        while (true) {
            out.println("Position");
            out.println(position);
            out.println(cell + "'s move");
            out.println("Enter row and column");
            int row = sc.scanParameter("> row = ");
            int column = sc.scanParameter("> column = ");
            final Move move = new Move(row, column, cell);
            if (position.isValid(move)) {
                return move;
            }
            out.println("Move " + move + " is invalid. Try again.");
        }
    }
}
