package md2html.parser;

/**
 * @author Georgiy Korneev (kgeorgiy@kgeorgiy.info)
 */
public interface CharSource {
    boolean hasNext();

    boolean hasNext(int count);

    char previous();

    char next();

    char next(int shift);

    int getPosition();
}
