package headers;

import java.util.regex.*;
import java.util.Map;
import java.util.Iterator;

public class Lexer implements Iterator<Token>, Iterable<Token> {
    private final Map<Token.Type, Pattern> patterns = Map.ofEntries(
            Map.entry(Token.Type.Identifier, Pattern.compile("[a-zA-Z_:][0-9a-zA-Z_:]*")),
            Map.entry(Token.Type.Asterisk, Pattern.compile("\\*")),
            Map.entry(Token.Type.OpeningParenthesis, Pattern.compile("\\(")),
            Map.entry(Token.Type.ClosingParenthesis, Pattern.compile("\\)")),
            Map.entry(Token.Type.Comma, Pattern.compile(","))
    );

    private final Pattern skip = Pattern.compile("[ \t\n\r]+");
    private final Matcher matcher = skip.matcher("");

    private int begin = 0;
    private int end = 0;
    private boolean hasNext = true;

    private final String text;

    public int position() {
        return end;
    }

    @Override
    public Iterator<Token> iterator() {
        return this;
    }

    public Lexer(String text) {
        this.text = text;
    }

    @Override
    public boolean hasNext() {
        return hasNext;
    }

    private boolean matchLookingAt() {
        if (matcher.lookingAt()) {
            begin = end;
            end = begin + matcher.end();
            matcher.reset(text.substring(end));
            return true;
        }
        return false;
    }

    @Override
    public Token next() {
        begin = end;
        matcher.usePattern(skip);
        matcher.reset(text.substring(begin));
        matchLookingAt();
        for (final Token.Type type : Token.Type.values()) {
            if (type == Token.Type.END || type == Token.Type.Epsilon) {
                continue;
            }
            matcher.usePattern(patterns.get(type));
            if (matchLookingAt()) {
                return new Token(type, text.substring(begin, end));
            }
        }
        if (end != text.length()) {
            throw new Error();
        }
        hasNext = false;
        return new Token(Token.Type.END, null);
    }
}