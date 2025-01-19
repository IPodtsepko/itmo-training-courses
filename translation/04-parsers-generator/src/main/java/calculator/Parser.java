package calculator;



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
     * A class containing information for the rule "expression".
     */
    public static class ExpressionContext extends Node {
        public ExpressionContext(String name) {
            super(name);
        }

        /// Synthesized attributes:
        public Integer result;
        public Integer external;

        /// Inherited attributes:

    }


    /**
     * A function that performs parsing according to the rule "expression".
     */
    public ExpressionContext expression() throws ParseException {
        // Attributes declarations
        Integer result = null;
        Integer external = null;
        // Variables from rules
        TermContext term = null;
        TermsContext terms = null;
        var _localctx = new ExpressionContext("expression");

        switch (curToken.token) {
            case OpeningParenthesis, Natural -> {
                // Start of processing parsing rule
                term = term();
                _localctx.children.add(term);
                // End of processing parsing rule

                // Start of code insertion
                result = term.result; external = term.result;;
                // End of code insertion

                // Start of processing parsing rule
                terms = terms(external);
                _localctx.children.add(terms);
                // End of processing parsing rule

                // Start of code insertion
                result = terms.result; external = result;;
                // End of code insertion

            }
            default -> throw new ParseException("Unexpected token", lexer.position());
        }
        // Wrapping inherited attributes

        // Wrapping synthesized attributes
        _localctx.result = result;
        _localctx.external = external;

        return _localctx;

    }


    /**
     * A class containing information for the rule "term".
     */
    public static class TermContext extends Node {
        public TermContext(String name) {
            super(name);
        }

        /// Synthesized attributes:
        public Integer result;
        public Integer internal;

        /// Inherited attributes:

    }


    /**
     * A function that performs parsing according to the rule "term".
     */
    public TermContext term() throws ParseException {
        // Attributes declarations
        Integer result = null;
        Integer internal = null;
        // Variables from rules
        MultiplierContext multiplier = null;
        MultipliersContext multipliers = null;
        var _localctx = new TermContext("term");

        switch (curToken.token) {
            case OpeningParenthesis, Natural -> {
                // Start of processing parsing rule
                multiplier = multiplier();
                _localctx.children.add(multiplier);
                // End of processing parsing rule

                // Start of code insertion
                internal = multiplier.result;;
                // End of code insertion

                // Start of processing parsing rule
                multipliers = multipliers(internal);
                _localctx.children.add(multipliers);
                // End of processing parsing rule

                // Start of code insertion
                result = multipliers.result;;
                // End of code insertion

            }
            default -> throw new ParseException("Unexpected token", lexer.position());
        }
        // Wrapping inherited attributes

        // Wrapping synthesized attributes
        _localctx.result = result;
        _localctx.internal = internal;

        return _localctx;

    }


    /**
     * A class containing information for the rule "terms".
     */
    public static class TermsContext extends Node {
        public TermsContext(String name) {
            super(name);
        }

        /// Synthesized attributes:
        public Integer result;
        public Integer internal;

        /// Inherited attributes:
        public Integer external;
    }


    /**
     * A function that performs parsing according to the rule "terms".
     */
    public TermsContext terms(Integer external) throws ParseException {
        // Attributes declarations
        Integer result = null;
        Integer internal = null;
        // Variables from rules
        TermContext term = null;
        Token Plus = null;
        TermsContext terms = null;
        Token Minus = null;
        Token Epsilon = null;
        var _localctx = new TermsContext("terms");

        switch (curToken.token) {
            case Plus -> {
                // Start of processing lexer rule
                assertToken(Token.Type.Plus);
                Plus = curToken;
                _localctx.children.add(new Node("Plus:" + curToken.text));
                curToken = lexer.next();
                // End of processing lexer rule

                // Start of processing parsing rule
                term = term();
                _localctx.children.add(term);
                // End of processing parsing rule

                // Start of code insertion
                internal = external + term.result;;
                // End of code insertion

                // Start of processing parsing rule
                terms = terms(internal);
                _localctx.children.add(terms);
                // End of processing parsing rule

                // Start of code insertion
                result = terms.result;;
                // End of code insertion

            }
            case Minus -> {
                // Start of processing lexer rule
                assertToken(Token.Type.Minus);
                Minus = curToken;
                _localctx.children.add(new Node("Minus:" + curToken.text));
                curToken = lexer.next();
                // End of processing lexer rule

                // Start of processing parsing rule
                term = term();
                _localctx.children.add(term);
                // End of processing parsing rule

                // Start of code insertion
                internal = external - term.result;;
                // End of code insertion

                // Start of processing parsing rule
                terms = terms(internal);
                _localctx.children.add(terms);
                // End of processing parsing rule

                // Start of code insertion
                result = terms.result;;
                // End of code insertion

            }
            case END, ClosingParenthesis -> {
                _localctx.children.add(new Node("Epsilon"));
                // Start of code insertion
                result = external;;
                // End of code insertion

            }
            default -> throw new ParseException("Unexpected token", lexer.position());
        }
        // Wrapping inherited attributes
        _localctx.external = external;
        // Wrapping synthesized attributes
        _localctx.result = result;
        _localctx.internal = internal;

        return _localctx;

    }


    /**
     * A class containing information for the rule "multiplier".
     */
    public static class MultiplierContext extends Node {
        public MultiplierContext(String name) {
            super(name);
        }

        /// Synthesized attributes:
        public Integer result;

        /// Inherited attributes:

    }


    /**
     * A function that performs parsing according to the rule "multiplier".
     */
    public MultiplierContext multiplier() throws ParseException {
        // Attributes declarations
        Integer result = null;
        // Variables from rules
        MaybeFactorialContext maybeFactorial = null;
        ArgumentContext argument = null;
        var _localctx = new MultiplierContext("multiplier");

        switch (curToken.token) {
            case OpeningParenthesis, Natural -> {
                // Start of processing parsing rule
                argument = argument();
                _localctx.children.add(argument);
                // End of processing parsing rule

                // Start of code insertion
                result = argument.result;;
                // End of code insertion

                // Start of processing parsing rule
                maybeFactorial = maybeFactorial(result);
                _localctx.children.add(maybeFactorial);
                // End of processing parsing rule

                // Start of code insertion
                result = maybeFactorial.result;
                // End of code insertion

            }
            default -> throw new ParseException("Unexpected token", lexer.position());
        }
        // Wrapping inherited attributes

        // Wrapping synthesized attributes
        _localctx.result = result;

        return _localctx;

    }


    /**
     * A class containing information for the rule "argument".
     */
    public static class ArgumentContext extends Node {
        public ArgumentContext(String name) {
            super(name);
        }

        /// Synthesized attributes:
        public Integer result;

        /// Inherited attributes:

    }


    /**
     * A function that performs parsing according to the rule "argument".
     */
    public ArgumentContext argument() throws ParseException {
        // Attributes declarations
        Integer result = null;
        // Variables from rules
        ExpressionContext expression = null;
        Token Natural = null;
        Token ClosingParenthesis = null;
        Token OpeningParenthesis = null;
        var _localctx = new ArgumentContext("argument");

        switch (curToken.token) {
            case OpeningParenthesis -> {
                // Start of processing lexer rule
                assertToken(Token.Type.OpeningParenthesis);
                OpeningParenthesis = curToken;
                _localctx.children.add(new Node("OpeningParenthesis:" + curToken.text));
                curToken = lexer.next();
                // End of processing lexer rule

                // Start of processing parsing rule
                expression = expression();
                _localctx.children.add(expression);
                // End of processing parsing rule

                // Start of code insertion
                result = expression.result;;
                // End of code insertion

                // Start of processing lexer rule
                assertToken(Token.Type.ClosingParenthesis);
                ClosingParenthesis = curToken;
                _localctx.children.add(new Node("ClosingParenthesis:" + curToken.text));
                curToken = lexer.next();
                // End of processing lexer rule

            }
            case Natural -> {
                // Start of processing lexer rule
                assertToken(Token.Type.Natural);
                Natural = curToken;
                _localctx.children.add(new Node("Natural:" + curToken.text));
                curToken = lexer.next();
                // End of processing lexer rule

                // Start of code insertion
                result = Integer.parseInt(Natural.text);;
                // End of code insertion

            }
            default -> throw new ParseException("Unexpected token", lexer.position());
        }
        // Wrapping inherited attributes

        // Wrapping synthesized attributes
        _localctx.result = result;

        return _localctx;

    }


    /**
     * A class containing information for the rule "maybeFactorial".
     */
    public static class MaybeFactorialContext extends Node {
        public MaybeFactorialContext(String name) {
            super(name);
        }

        /// Synthesized attributes:
        public Integer result;

        /// Inherited attributes:
        public Integer value;
    }


    /**
     * A function that performs parsing according to the rule "maybeFactorial".
     */
    public MaybeFactorialContext maybeFactorial(Integer value) throws ParseException {
        // Attributes declarations
        Integer result = null;
        // Variables from rules
        Token Factorial = null;
        Token Epsilon = null;
        Token SubFactorial = null;
        var _localctx = new MaybeFactorialContext("maybeFactorial");

        switch (curToken.token) {
            case SubFactorial -> {
                // Start of processing lexer rule
                assertToken(Token.Type.SubFactorial);
                SubFactorial = curToken;
                _localctx.children.add(new Node("SubFactorial:" + curToken.text));
                curToken = lexer.next();
                // End of processing lexer rule

                // Start of code insertion
                
                result = 1;
                while (value > 1) {
                    result *= value;
                    value -= 2;
                };
                // End of code insertion

            }
            case Factorial -> {
                // Start of processing lexer rule
                assertToken(Token.Type.Factorial);
                Factorial = curToken;
                _localctx.children.add(new Node("Factorial:" + curToken.text));
                curToken = lexer.next();
                // End of processing lexer rule

                // Start of code insertion
                
                result = 1;
                while (value > 1) {
                    result *= value;
                    value -= 1;
                };
                // End of code insertion

            }
            case Multiplication, Division, END, ClosingParenthesis, Plus, Minus -> {
                _localctx.children.add(new Node("Epsilon"));
                // Start of code insertion
                result = value;;
                // End of code insertion

            }
            default -> throw new ParseException("Unexpected token", lexer.position());
        }
        // Wrapping inherited attributes
        _localctx.value = value;
        // Wrapping synthesized attributes
        _localctx.result = result;

        return _localctx;

    }


    /**
     * A class containing information for the rule "multipliers".
     */
    public static class MultipliersContext extends Node {
        public MultipliersContext(String name) {
            super(name);
        }

        /// Synthesized attributes:
        public Integer result;
        public Integer internal;

        /// Inherited attributes:
        public Integer external;
    }


    /**
     * A function that performs parsing according to the rule "multipliers".
     */
    public MultipliersContext multipliers(Integer external) throws ParseException {
        // Attributes declarations
        Integer result = null;
        Integer internal = null;
        // Variables from rules
        Token Epsilon = null;
        Token Division = null;
        Token Multiplication = null;
        MultiplierContext multiplier = null;
        MultipliersContext multipliers = null;
        var _localctx = new MultipliersContext("multipliers");

        switch (curToken.token) {
            case Multiplication -> {
                // Start of processing lexer rule
                assertToken(Token.Type.Multiplication);
                Multiplication = curToken;
                _localctx.children.add(new Node("Multiplication:" + curToken.text));
                curToken = lexer.next();
                // End of processing lexer rule

                // Start of processing parsing rule
                multiplier = multiplier();
                _localctx.children.add(multiplier);
                // End of processing parsing rule

                // Start of code insertion
                internal = external * multiplier.result;;
                // End of code insertion

                // Start of processing parsing rule
                multipliers = multipliers(internal);
                _localctx.children.add(multipliers);
                // End of processing parsing rule

                // Start of code insertion
                result = multipliers.result;;
                // End of code insertion

            }
            case Division -> {
                // Start of processing lexer rule
                assertToken(Token.Type.Division);
                Division = curToken;
                _localctx.children.add(new Node("Division:" + curToken.text));
                curToken = lexer.next();
                // End of processing lexer rule

                // Start of processing parsing rule
                multiplier = multiplier();
                _localctx.children.add(multiplier);
                // End of processing parsing rule

                // Start of code insertion
                internal = external / multiplier.result;;
                // End of code insertion

                // Start of processing parsing rule
                multipliers = multipliers(internal);
                _localctx.children.add(multipliers);
                // End of processing parsing rule

                // Start of code insertion
                result = multipliers.result;;
                // End of code insertion

            }
            case END, ClosingParenthesis, Plus, Minus -> {
                _localctx.children.add(new Node("Epsilon"));
                // Start of code insertion
                result = external;;
                // End of code insertion

            }
            default -> throw new ParseException("Unexpected token", lexer.position());
        }
        // Wrapping inherited attributes
        _localctx.external = external;
        // Wrapping synthesized attributes
        _localctx.result = result;
        _localctx.internal = internal;

        return _localctx;

    }



    public static class Node extends Tree {
        public Node(String name) {
            super(name);
        }
    }
}