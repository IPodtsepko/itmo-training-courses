package game;

import java.util.Arrays;
import java.util.Map;

public class MnkBoard implements Board {
    public static boolean isValidParameters(int m, int n, int k) {
        return 1 < n && 1 < m && 1 < k && k <= Integer.max(n, m);
    }

    protected static final Map<Cell, Character> SYMBOLS = Map.of(
            Cell.X, 'X',
            Cell.O, 'O',
            Cell.Y, '-',
            Cell.Z, '|',
            Cell.E, '.',
            Cell.WALL, ' '
    );
    private final int m;
    private final int n;
    private final int k;
    private final int playersCounter;
    protected final Cell[][] cells;
    protected int emptyCellCounter;
    private Cell turn;
    private final int firstColumnWidth;
    private final int columnWidth;

    private final Position position = new Position() {
        @Override
        public boolean isValid(Move move) {
            return move != null
                    && inBoard(move.getRow(), move.getColumn())
                    && cells[move.getRow()][move.getColumn()] == Cell.E
                    && move.getValue() == turn;
        }

        @Override
        public int getN() {
            return n;
        }

        @Override
        public int getM() {
            return m;
        }

        @Override
        public String toString() {
            return MnkBoard.this.toString();
        }
    };

    public MnkBoard(int m, int n, int k, int playersCounter) {
        if (!isValidParameters(m, n, k)) {
            throw new IllegalArgumentException("Invalid parameters of board");
        }
        this.m = m;
        this.n = n;
        this.k = k;
        this.playersCounter = playersCounter;
        emptyCellCounter = n * m;
        firstColumnWidth = Integer.toString(m - 1).length() + 1;
        columnWidth = Integer.toString(n - 1).length() + 1;
        cells = new Cell[n][m];
        FillBoard();
        turn = Cell.X;
    }

    protected void FillBoard() {
        for (Cell[] row : cells) {
            Arrays.fill(row, Cell.E);
        }
    }

    @Override
    public Position getPosition() {
        return position;
    }

    @Override
    public Result makeMove(final Move move) {
        if (!position.isValid(move)) { return Result.LOSE; }

        int r = move.getRow();
        int c = move.getColumn();
        cells[r][c] = move.getValue();
        emptyCellCounter--;

        if (isWin(move, 1, 0) || isWin(move, 0,1)
                || isWin(move, 1, 1) || isWin(move, 1, -1)) {
            return Result.WIN;
        }

        if (emptyCellCounter == 0) { return Result.DRAW; }

        turn = nextCell();
        return Result.UNKNOWN;
    }

    private boolean isWin(Move move, int dR, int dC) {
        int r = move.getRow();
        int c = move.getColumn();
        return getCountOfCells(r, c, dR, dC) + getCountOfCells(r, c, -dR, -dC) > k;
    }

    private int getCountOfCells(int r, int c, int dR, int dC) {
        Cell cell = cells[r][c];
        int count = 0;
        while (inBoard(r, c) && cells[r][c] == cell) {
            count++;
            r += dR;
            c += dC;
        }
        return count;
    }

    private boolean inBoard(int r, int c) {
        return -1 < r && r < n && -1 < c && c < m;
    }

    @Override
    public Cell getCell() {
        return turn;
    }

    private Cell nextCell() {
        int current = Cell.LIST_OF_PLAYER_CELLS.indexOf(turn);
        return Cell.LIST_OF_PLAYER_CELLS.get((current + 1) % playersCounter);
    }

    @Override
    public String toString() {
        final StringBuilder sb = new StringBuilder();
        fillColumn(sb, "", firstColumnWidth);
        for (int c = 0; c < m; c++) {
            fillColumn(sb, Integer.toString(c), columnWidth);
        }
        for (int r = 0; r < n; r++) {
            sb.append("\n");
            fillColumn(sb, Integer.toString(r), firstColumnWidth);
            for (int c = 0; c < m; c++) {
                fillColumn(sb, SYMBOLS.get(cells[r][c]).toString(), columnWidth);
            }
        }
        return sb.toString();
    }

    private void fillColumn(StringBuilder dest, String data, int columnWidth) {
        dest.append(data);
        for (int i = data.length(); i < columnWidth; i++) {
            dest.append(" ");
        }
    }

    public int getPlayersCounter() {
        return playersCounter;
    }
}
