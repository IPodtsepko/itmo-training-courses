package game;

import java.util.List;

public class Game {
    private final boolean log;
    private final List<Player> players;

    public Game(final boolean log, final List<Player> players) {
        this.log = log;
        this.players = players;
    }

    public int play(final Board board) {
        if (players.size() != board.getPlayersCounter()) {
            throw new AssertionError("The number of players on the board is" +
                    "different from the number of players in the game.");
        }
        while (true) {
            for (int i = 0; i < players.size(); i++) {
                final int result = move(board, players.get(i), i + 1);
                if (result != -1) {
                    return result;
                }
            }
        }
    }

    private int move(final Board board, final Player player, final int no) {
        final Move move = player.move(board.getPosition(), board.getCell());
        final Result result = board.makeMove(move);
        log(String.format("Player %d move: %s", no, move));
        log("Position:\n" + board);
        if (result == Result.WIN) {
            log(String.format("Player %d won", no));
            return no;
        } else if (result == Result.LOSE) {
            log(String.format("Player %d lose", no));
            return -2;
        } else if (result == Result.DRAW) {
            log("Draw");
            return 0;
        } else {
            return -1;
        }
    }

    private void log(final String message) {
        if (log) {
            System.out.println(message);
        }
    }
}
