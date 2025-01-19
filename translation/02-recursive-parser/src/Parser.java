import java.io.InputStream;
import java.text.ParseException;

/**
 * @author Игорь Подцепко (i.podtsepko2002@gmail.com
 */
public class Parser {
    LexicalAnalyzer analyzer;

    public Parser(InputStream inputStream) throws ParseException {
        analyzer = new LexicalAnalyzer(inputStream);
        analyzer.nextToken();
    }

    /**
     * {@code F -> T N(A);}
     */
    Tree parseF() throws ParseException {
        if (analyzer.getCurrentToken() != Token.WORD) {
            throw new AssertionError();
        }

        var T = parseT();
        var N = parseN();
        var openingBracket = expectTerminal(Token.OPENING_BRACKET);
        var A = parseA();
        var closingBracket = expectTerminal(Token.CLOSING_BRACKET);
        var semicolon = expectTerminal(Token.SEMICOLON);

        return new Tree("F", T, N, openingBracket, A, closingBracket, semicolon);
    }

    Tree expectTerminal(final Token token) throws ParseException {
        var name = analyzer.getTokenName(token);
        if (analyzer.getCurrentToken() != token) {
            throw analyzer.newException();
        }
        var result = new Tree(name);
        analyzer.nextToken();
        return result;
    }

    /**
     * {@code T -> NP}
     */
    Tree parseT() throws ParseException {
        var N = parseN();
        var P = parseP();
        return new Tree("T", N, P);
    }

    /**
     * {@code P -> *P|ε|&}
     */
    Tree parseP() throws ParseException {
        return switch (analyzer.getCurrentToken()) {
            case WORD -> // P -> ε
                    new Tree("P", new Tree("ε"));
            case REFERENCE -> { // P -> &
                analyzer.nextToken();
                yield new Tree("P", new Tree("&"));
            }
            case STAR -> { // P -> *P
                analyzer.nextToken();
                yield new Tree("P", new Tree("*"), parseP());
            }
            default -> throw analyzer.newException();
        };
    }

    /**
     * {@code A -> T NB|ε}
     */
    Tree parseA() throws ParseException {
        return switch (analyzer.getCurrentToken()) {
            case CLOSING_BRACKET -> // A -> ε
                    new Tree("A", new Tree("ε"));
            case WORD -> { // A -> T NB
                var T = parseT();
                var N = parseN();
                var B = parseB();
                yield new Tree("A", T, N, B);
            }
            default -> throw analyzer.newException();
        };
    }

    /**
     * {@code B -> , T NB|ε}
     */
    Tree parseB() throws ParseException {
        return switch (analyzer.getCurrentToken()) {
            case CLOSING_BRACKET -> // B -> ε
                    new Tree("B", new Tree("ε"));
            case COMMA -> { // B -> , T NB
                var comma = expectTerminal(Token.COMMA);
                var T = parseT();
                var N = parseN();
                var B = parseB();
                yield new Tree("B", comma, T, N, B);
            }
            default -> throw analyzer.newException();
        };
    }

    /**
     * {@code N -> w}
     */
    Tree parseN() throws ParseException {
        if (analyzer.getCurrentToken() != Token.WORD) {
            throw analyzer.newException();
        }
        var N = new Tree("N", new Tree(analyzer.getWord()));
        analyzer.nextToken();
        return N;
    }
}
