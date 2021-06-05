package game;

import java.util.List;

public enum Cell {
    X, O, Y, Z, E, WALL;
    public static final List<Cell> LIST_OF_PLAYER_CELLS = List.of(X, O, Y, Z);
}
