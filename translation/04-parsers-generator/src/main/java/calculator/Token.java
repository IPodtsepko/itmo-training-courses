package calculator;

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
        SubFactorial,
        Factorial,
        ClosingParenthesis,
        Division,
        Minus,
        Multiplication,
        Natural,
        OpeningParenthesis,
        Plus,
        END,
        Epsilon
    }

    @Override
    public String toString() {
        return text;
    }
}
