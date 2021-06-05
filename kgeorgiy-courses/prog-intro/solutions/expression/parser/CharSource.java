package expression.parser;

/**
 * @author Georgiy Korneev (kgeorgiy@kgeorgiy.info)
 */
public interface CharSource {
    boolean hasNext();

    boolean hasNext(int count);

    char next();

    char next(int shift);

    int getPosition();

    ParseException error(final String message);
}
