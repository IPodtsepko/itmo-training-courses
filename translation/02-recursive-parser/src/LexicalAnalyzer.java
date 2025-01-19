import java.io.IOException;
import java.io.InputStream;
import java.text.ParseException;


/**
 * @author Игорь Подцепко (i.podtsepko2002@gmail.com
 */
public class LexicalAnalyzer {
    final InputStream inputStream;
    int currentChar;
    int currentPosition;
    String word;
    Token currentToken;

    public LexicalAnalyzer(final InputStream inputStream) throws ParseException {
        this.inputStream = inputStream;
        currentPosition = 0;
        nextChar();
    }

    private char nextChar() throws ParseException {
        currentPosition++;
        try {
            currentChar = inputStream.read();
            return (char) currentChar;
        } catch (IOException e) {
            throw new ParseException(e.getLocalizedMessage(), currentPosition);
        }
    }

    public void nextToken() throws ParseException {
        skipWhitespaces();
        if (Character.isLetter(currentChar) || currentChar == '_') {
            parseWord();
            return;
        }
        currentToken = switch (currentChar) {
            case '*' -> Token.STAR;
            case '(' -> Token.OPENING_BRACKET;
            case ',' -> Token.COMMA;
            case ')' -> Token.CLOSING_BRACKET;
            case ';' -> Token.SEMICOLON;
            case '&' -> Token.REFERENCE;
            case -1 -> Token.END;
            default ->  throw newException();
        };

        nextChar();
    }

    private void skipWhitespaces() throws ParseException {
        while (Character.isWhitespace(currentChar)) {
            nextChar();
        }
    }

    private void parseWord() throws ParseException {
        final StringBuilder word = new StringBuilder();
        word.append((char) currentChar);
        nextChar();
        while (Character.isLetterOrDigit(currentChar) || currentChar == '_') {
            word.append((char) currentChar);
            nextChar();
        }
        this.word = word.toString();
        currentToken = Token.WORD;
    }

    public Token getCurrentToken() {
        return currentToken;
    }
    public String getWord() {
        return word;
    }
    public String getTokenName(final Token token) {
        return switch (token) {
            case STAR -> "*";
            case OPENING_BRACKET -> "(";
            case COMMA -> ",";
            case CLOSING_BRACKET -> ")";
            case SEMICOLON -> ";";
            case WORD -> word;
            case END -> "EOF";
            case REFERENCE -> "&";
        };
    }

    public ParseException newException() {
        return new ParseException("Unexpected token", currentPosition);
    }
}
