package game;

import java.util.ArrayList;
import java.util.List;

public class Main {
    public static void main(String[] args) {
        IntParameterScanner sc = new IntParameterScanner(new Scanner(System.in), System.out);
        Board board;

        System.out.println("Enter count of players (2, 3 or 4):");
        int playersCounter = sc.scanParameter(
                "> count of players = ",
                "Count of players must be 2, 3 or 4. Try again.",
                x -> 2 <= x && x <= 4
        );

        System.out.println("\nChoose board:\n1. m,n,k-board\n2. rhombus");
        int boardNumber = sc.scanParameter(
                "> board ID = ",
                "Board ID must be 1 or 2. Try again.",
                x -> 1 <= x && x <= 2);

        System.out.println("\nEnter board parameters:");
        switch (boardNumber) {
            case 1:
                board = scanMnkBoard(sc, playersCounter);
                break;
            case 2:
                board = scanRhombusBoard(sc, playersCounter);
                break;
            default:
                throw new IllegalStateException("Unexpected value: " + boardNumber);
        }
        System.out.println("\nChoose players type:\n1. HumanPlayer\n2. SequentialPlayer\n3. RandomPlayer");
        List<Player> players = scanPlayers(sc, playersCounter);

        final Game game = new Game(true, players);
        int result;
        do {
            result = game.play(board);
            System.out.println("Game result: " + result);
        } while (result < 1);
    }

    private static MnkBoard scanMnkBoard(final IntParameterScanner sc, int playersCounter) {
        while (true) {
            int m = sc.scanParameter("> m = ");
            int n = sc.scanParameter("> n = ");
            int k = sc.scanParameter("> k = ");
            if (MnkBoard.isValidParameters(m, n, k)) {
                return new MnkBoard(m, n, k, playersCounter);
            }
            System.out.println("Is invalid m,n,k-board parameters. Try again.");
        }
    }

    private static RhombusBoard scanRhombusBoard(final IntParameterScanner sc, int playersCounter) {
        while (true) {
            int side = sc.scanParameter("> side = ");
            int k = sc.scanParameter("> k = ");
            if (RhombusBoard.isValidParameters(side, k)) {
                return new RhombusBoard(side, k, playersCounter);
            }
            System.out.println("Is invalid rhombus board parameters. Try again.");
        }
    }

    private static List<Player> scanPlayers(final IntParameterScanner sc, int playersCounter) {
        List<Player> players = new ArrayList<>(playersCounter);
        players.add(new HumanPlayer());
        for (int i = 2; i < playersCounter + 1; i++) {

            int playerType = sc.scanParameter(
                    String.format("> type of player %d = ", i),
                    "Type of player must be 1, 2 or 3. Try again.",
                    x -> 1 <= x && x <= 3
            );
            switch (playerType) {
                case 1:
                    players.add(new HumanPlayer());
                    break;
                case 2:
                    players.add(new SequentialPlayer());
                    break;
                case 3:
                    players.add(new RandomPlayer());
                    break;
            }
        }
        return players;
    }
}