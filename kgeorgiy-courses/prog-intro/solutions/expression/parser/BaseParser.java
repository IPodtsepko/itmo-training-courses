package expression.parser;

import expression.exceptions.parse_exceptions.UnexpectedCharException;

/**
 * @author Georgiy Korneev (kgeorgiy@kgeorgiy.info)
 */
public class BaseParser {
    public static final char END = '\0';
    protected char ch = 0xffff;
    private CharSource source;

    protected void setSource(final CharSource source) {
        this.source = source;
        nextChar();
    }

    protected void nextChar() {
        ch = source.hasNext() ? source.next() : END;
    }

    protected boolean test(char expected) {
        if (ch == expected) {
            nextChar();
            return true;
        }
        return false;
    }

    protected boolean test(String expected) {
        if (!source.hasNext(expected.length())) {
            return false;
        }
        for (int i = 0; i < expected.length(); i++) {
            if (expected.charAt(i) != source.next(i)) {
                return false;
            }
        }
        expect(expected);
        return true;
    }

    protected void expect(final char c) {
        if (ch != c) {
            throw new UnexpectedCharException(c, ch, getPosition());
        }
        nextChar();
    }

    protected void expect(final String value) {
        for (char c : value.toCharArray()) {
            expect(c);
        }
    }

    protected int getPosition() {
        return source.getPosition();
    }

    protected boolean eof() {
        return test(END);
    }

    protected ParseException error(final String message) {
        return source.error(message);
    }

    protected boolean between(final char from, final char to) {
        return from <= ch && ch <= to;
    }
}
