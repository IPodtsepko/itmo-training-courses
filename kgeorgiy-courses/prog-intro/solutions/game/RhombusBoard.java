package game;

public class RhombusBoard extends MnkBoard {
    public static boolean isValidParameters(int side, int k) {
        return MnkBoard.isValidParameters(2 * side - 1, 2 * side - 1, k);
    }

    public RhombusBoard(int side, int k, int playersCounter) {
        super(2 * side - 1, 2 * side - 1, k, playersCounter);
    }

    @Override
    protected void FillBoard() {
        int d = getPosition().getN();
        int wellCellCounter = d % 2 < 1 ? (d - 2) / 2 : (d - 1) / 2;
        for (int r = 0; r < d; r++) {
            for (int c = 0; c < wellCellCounter; c++) {
                cells[r][c] = Cell.WALL;
                cells[r][(d - 1) - c] = Cell.WALL;
            }
            for (int c = wellCellCounter; c < d - wellCellCounter; c++) {
                cells[r][c] = Cell.E;
            }
            emptyCellCounter -= 2 * wellCellCounter;
            if (r < d / 2) {
                wellCellCounter = Integer.max(wellCellCounter - 1, 0);
            } else {
                wellCellCounter += 1;
            }
        }
    }
}
