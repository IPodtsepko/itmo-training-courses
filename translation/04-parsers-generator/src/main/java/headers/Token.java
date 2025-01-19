package headers;

/**
 * A class containing the necessary information about the token.
 * <p>
 * Note: This file is generated automatically.
 * </p>
 */
public class Token {
    public String text;
    public Type token;

    public Token(Type token, String text) {
        this.text = text;
        this.token = token;
    }

    enum Type {
        Identifier,
        Asterisk,
        OpeningParenthesis,
        ClosingParenthesis,
        Comma,
        END,
        Epsilon
    }

    @Override
    public String toString() {
        return text;
    }
}
