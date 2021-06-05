package game;

public interface Position {
    boolean isValid(Move move);

    int getN();

    int getM();
}
