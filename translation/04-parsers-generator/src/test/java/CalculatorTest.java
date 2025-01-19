import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

import calculator.Parser;

import java.text.ParseException;

public class CalculatorTest {
    void assertResult(final int expected, final String expression) throws ParseException {
        final Parser parser = new Parser(expression);
        final Parser.ExpressionContext ctx = parser.expression();
        assertEquals(expected, ctx.result, String.format("Invalid evaluation result of '%s'", expression));
    }

    @Test
    void add() throws ParseException {
        assertResult(2, "1 + 1");
        assertResult(1, "1 + 0");
        assertResult(4, "1 + 1 + 2");
    }

    @Test
    void multiply() throws ParseException {
        assertResult(0, "10 * 0");
        assertResult(0, "0*10");
        assertResult(10, "1*        10");
        assertResult(10, "10                         *\t1");
        assertResult(6, "2 * 3");
    }

    @Test
    void subtraction() throws ParseException {
        assertResult(-1, "0 - 1");
        assertResult(0, "0 - 0");
        assertResult(3, "9 - 6");
        assertResult(1, "10 - 1-2-3-3");
    }

    @Test
    void division() throws ParseException {
        assertResult(0, "1 / 2");
        assertResult(2, "10/5");
        assertResult(8, "32/2/2");
    }

    @Test
    void brackets() throws ParseException {
        assertResult(0, "(( ((((( (0 ))  ))))))   ");
        assertResult(2, "(((  (((  1)))) + ((1))))");
        assertResult(6, "((((2 * (((3))))       ) ) )");
        assertResult(2, "1 - (2 - 3)");
        assertResult(32, "32/(2 / 2)");
    }

    @Test
    void allOperations() throws ParseException {
        assertResult(0, "1 + 2 - 3");
        assertResult(6, "2 + 2 * 2");
        assertResult(10, "4 * 3-2 * 1");
        assertResult(4, "4 * (3-2) * 1");
        assertResult(2, "(1+2 * 3 - 3+4) - 2 - 2-2");
        assertResult(48, "99 / (11 * 3 - 22) / 3*16");
    }

    @Test
    void modification() throws ParseException {
        assertResult(1, "0!");
        assertResult(1, "1!");
        assertResult(2, "2!");
        assertResult(6, "3!");
        assertResult(24, "4!");
        assertResult(120, "5!");

        assertResult(1, "0!!");
        assertResult(1, "1!!");
        assertResult(2, "2!!");
        assertResult(3, "3!!");
        assertResult(8, "4!!");
        assertResult(15, "5!!");

        assertResult(12, "((1 + 2)!!)! * 2");

        assertResult(0, "3!!!!!");
    }
}
