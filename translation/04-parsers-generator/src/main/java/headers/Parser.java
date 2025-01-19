package headers;



import java.text.ParseException;
import org.generator.util.Tree;

public class Parser {
    public Token curToken;
    public final Lexer lexer;

    public Parser(String text) {
        lexer = new Lexer(text);
        curToken = lexer.next();
    }

    public void assertToken(final Token.Type expected) throws ParseException {
        if (expected != curToken.token) {
            throw new ParseException("Unexpected token at %s", lexer.position());
        }
    }

    /**
     * A class containing information for the rule "functionHeader".
     */
    public static class FunctionHeaderContext extends Node {
        public FunctionHeaderContext(String name) {
            super(name);
        }

        /// Synthesized attributes:


        /// Inherited attributes:

    }


    /**
     * A function that performs parsing according to the rule "functionHeader".
     */
    public FunctionHeaderContext functionHeader() throws ParseException {
        // Attributes declarations

        // Variables from rules
        IdentifierContext identifier = null;
        TypeContext type = null;
        Token ClosingParenthesis = null;
        Token OpeningParenthesis = null;
        ArgumentsContext arguments = null;
        var _localctx = new FunctionHeaderContext("functionHeader");

        switch (curToken.token) {
            case Identifier -> {
                // Start of processing parsing rule
                type = type();
                _localctx.children.add(type);
                // End of processing parsing rule

                // Start of processing parsing rule
                identifier = identifier();
                _localctx.children.add(identifier);
                // End of processing parsing rule

                // Start of processing lexer rule
                assertToken(Token.Type.OpeningParenthesis);
                OpeningParenthesis = curToken;
                _localctx.children.add(new Node("OpeningParenthesis:" + curToken.text));
                curToken = lexer.next();
                // End of processing lexer rule

                // Start of processing parsing rule
                arguments = arguments();
                _localctx.children.add(arguments);
                // End of processing parsing rule

                // Start of processing lexer rule
                assertToken(Token.Type.ClosingParenthesis);
                ClosingParenthesis = curToken;
                _localctx.children.add(new Node("ClosingParenthesis:" + curToken.text));
                curToken = lexer.next();
                // End of processing lexer rule

            }
            default -> throw new ParseException("Unexpected token", lexer.position());
        }
        // Wrapping inherited attributes

        // Wrapping synthesized attributes


        return _localctx;

    }


    /**
     * A class containing information for the rule "type".
     */
    public static class TypeContext extends Node {
        public TypeContext(String name) {
            super(name);
        }

        /// Synthesized attributes:


        /// Inherited attributes:

    }


    /**
     * A function that performs parsing according to the rule "type".
     */
    public TypeContext type() throws ParseException {
        // Attributes declarations

        // Variables from rules
        IdentifierContext identifier = null;
        PointersContext pointers = null;
        var _localctx = new TypeContext("type");

        switch (curToken.token) {
            case Identifier -> {
                // Start of processing parsing rule
                identifier = identifier();
                _localctx.children.add(identifier);
                // End of processing parsing rule

                // Start of processing parsing rule
                pointers = pointers();
                _localctx.children.add(pointers);
                // End of processing parsing rule

            }
            default -> throw new ParseException("Unexpected token", lexer.position());
        }
        // Wrapping inherited attributes

        // Wrapping synthesized attributes


        return _localctx;

    }


    /**
     * A class containing information for the rule "pointers".
     */
    public static class PointersContext extends Node {
        public PointersContext(String name) {
            super(name);
        }

        /// Synthesized attributes:


        /// Inherited attributes:

    }


    /**
     * A function that performs parsing according to the rule "pointers".
     */
    public PointersContext pointers() throws ParseException {
        // Attributes declarations

        // Variables from rules
        Token Epsilon = null;
        PointersContext pointers = null;
        Token Asterisk = null;
        var _localctx = new PointersContext("pointers");

        switch (curToken.token) {
            case Asterisk -> {
                // Start of processing lexer rule
                assertToken(Token.Type.Asterisk);
                Asterisk = curToken;
                _localctx.children.add(new Node("Asterisk:" + curToken.text));
                curToken = lexer.next();
                // End of processing lexer rule

                // Start of processing parsing rule
                pointers = pointers();
                _localctx.children.add(pointers);
                // End of processing parsing rule

            }
            case Identifier -> {
                _localctx.children.add(new Node("Epsilon"));
            }
            default -> throw new ParseException("Unexpected token", lexer.position());
        }
        // Wrapping inherited attributes

        // Wrapping synthesized attributes


        return _localctx;

    }


    /**
     * A class containing information for the rule "identifier".
     */
    public static class IdentifierContext extends Node {
        public IdentifierContext(String name) {
            super(name);
        }

        /// Synthesized attributes:


        /// Inherited attributes:

    }


    /**
     * A function that performs parsing according to the rule "identifier".
     */
    public IdentifierContext identifier() throws ParseException {
        // Attributes declarations

        // Variables from rules
        Token Identifier = null;
        var _localctx = new IdentifierContext("identifier");

        switch (curToken.token) {
            case Identifier -> {
                // Start of processing lexer rule
                assertToken(Token.Type.Identifier);
                Identifier = curToken;
                _localctx.children.add(new Node("Identifier:" + curToken.text));
                curToken = lexer.next();
                // End of processing lexer rule

            }
            default -> throw new ParseException("Unexpected token", lexer.position());
        }
        // Wrapping inherited attributes

        // Wrapping synthesized attributes


        return _localctx;

    }


    /**
     * A class containing information for the rule "arguments".
     */
    public static class ArgumentsContext extends Node {
        public ArgumentsContext(String name) {
            super(name);
        }

        /// Synthesized attributes:


        /// Inherited attributes:

    }


    /**
     * A function that performs parsing according to the rule "arguments".
     */
    public ArgumentsContext arguments() throws ParseException {
        // Attributes declarations

        // Variables from rules
        IdentifierContext identifier = null;
        Token Epsilon = null;
        TypeContext type = null;
        TailContext tail = null;
        var _localctx = new ArgumentsContext("arguments");

        switch (curToken.token) {
            case Identifier -> {
                // Start of processing parsing rule
                type = type();
                _localctx.children.add(type);
                // End of processing parsing rule

                // Start of processing parsing rule
                identifier = identifier();
                _localctx.children.add(identifier);
                // End of processing parsing rule

                // Start of processing parsing rule
                tail = tail();
                _localctx.children.add(tail);
                // End of processing parsing rule

            }
            case ClosingParenthesis -> {
                _localctx.children.add(new Node("Epsilon"));
            }
            default -> throw new ParseException("Unexpected token", lexer.position());
        }
        // Wrapping inherited attributes

        // Wrapping synthesized attributes


        return _localctx;

    }


    /**
     * A class containing information for the rule "tail".
     */
    public static class TailContext extends Node {
        public TailContext(String name) {
            super(name);
        }

        /// Synthesized attributes:


        /// Inherited attributes:

    }


    /**
     * A function that performs parsing according to the rule "tail".
     */
    public TailContext tail() throws ParseException {
        // Attributes declarations

        // Variables from rules
        IdentifierContext identifier = null;
        Token Comma = null;
        Token Epsilon = null;
        TypeContext type = null;
        TailContext tail = null;
        var _localctx = new TailContext("tail");

        switch (curToken.token) {
            case Comma -> {
                // Start of processing lexer rule
                assertToken(Token.Type.Comma);
                Comma = curToken;
                _localctx.children.add(new Node("Comma:" + curToken.text));
                curToken = lexer.next();
                // End of processing lexer rule

                // Start of processing parsing rule
                type = type();
                _localctx.children.add(type);
                // End of processing parsing rule

                // Start of processing parsing rule
                identifier = identifier();
                _localctx.children.add(identifier);
                // End of processing parsing rule

                // Start of processing parsing rule
                tail = tail();
                _localctx.children.add(tail);
                // End of processing parsing rule

            }
            case ClosingParenthesis -> {
                _localctx.children.add(new Node("Epsilon"));
            }
            default -> throw new ParseException("Unexpected token", lexer.position());
        }
        // Wrapping inherited attributes

        // Wrapping synthesized attributes


        return _localctx;

    }



    public static class Node extends Tree {
        public Node(String name) {
            super(name);
        }
    }
}