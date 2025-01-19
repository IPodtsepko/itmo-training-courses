import junit.framework.TestCase;

import java.io.ByteArrayInputStream;
import java.io.InputStream;
import java.text.ParseException;

public class ParserTest extends TestCase {

    @Override
    protected void setUp() throws Exception {
        Tree.resetId();
        super.setUp();
    }

    /**
     * Тест на все правила. Верхний уровень - {@code F -> T N(A);}.
     */
    public void testStrangeFunctionParseF() throws ParseException {
        String strangeFunction = "int * strangeFunction(Object x, char ** y);";
        InputStream stream = new ByteArrayInputStream(strangeFunction.getBytes());
        var parser = new Parser(stream);

        Tree F = parser.parseF();
        Tree.resetId();

        assertEquals(
                new Tree("F",
                        new Tree("T",
                                new Tree("N",
                                        new Tree("int")),
                                new Tree("P",
                                        new Tree("*"),
                                        new Tree("P",
                                                new Tree("ε")))),
                        new Tree("N",
                                new Tree("strangeFunction")),
                        new Tree("("),
                        new Tree("A",
                                new Tree("T",
                                        new Tree("N",
                                                new Tree("Object")),
                                        new Tree("P",
                                                new Tree("ε"))),
                                new Tree("N",
                                        new Tree("x")),
                                new Tree("B",
                                        new Tree(","),
                                        new Tree("T",
                                                new Tree("N",
                                                        new Tree("char")),
                                                new Tree("P",
                                                        new Tree("*"),
                                                        new Tree("P",
                                                                new Tree("*"),
                                                                new Tree("P",
                                                                        new Tree("ε"))))),
                                        new Tree("N",
                                                new Tree("y")),
                                        new Tree("B", new Tree("ε")))),
                        new Tree(")"),
                        new Tree(";")), F);
    }

    /**
     * Тест разбора типов. Правило {@code T -> NP}.
     */
    public void testWithPointerToObjectParseT() throws ParseException {
        String justWord = "Object * obj";
        InputStream stream = new ByteArrayInputStream(justWord.getBytes());
        var parser = new Parser(stream);

        Tree T = parser.parseT();
        Tree.resetId();

        assertEquals(
                new Tree("T",
                        new Tree("N",
                                new Tree("Object")),
                        new Tree("P",
                                new Tree("*"),
                                new Tree("P",
                                        new Tree("ε")))), T);
    }

    /**
     * Тест разбора указателей. Правило {@code P -> ε}.
     */
    public void testWithoutStartParseP() throws ParseException {
        String justWord = "variableName";
        InputStream stream = new ByteArrayInputStream(justWord.getBytes());
        var parser = new Parser(stream);

        Tree P = parser.parseP();
        Tree.resetId();

        assertEquals(new Tree("P", new Tree("ε")), P);
    }

    /**
     * Тест разбора указателей. Правило {@code P -> *P}.
     */
    public void testWithSomeStarsParseP() throws ParseException {
        String someStars = "**** variableName";
        InputStream stream = new ByteArrayInputStream(someStars.getBytes());
        var parser = new Parser(stream);

        Tree P = parser.parseP();
        Tree.resetId();

        assertEquals(
                new Tree("P",
                        new Tree("*"),
                        new Tree("P",
                                new Tree("*"),
                                new Tree("P",
                                        new Tree("*"),
                                        new Tree("P",
                                                new Tree("*"),
                                                new Tree("P",
                                                        new Tree("ε")))))), P);
    }

    /**
     * Тест разбора аргументов функции. Правило {@code A -> ε}.
     */
    public void testEmptyInputParseA() throws ParseException {
        String emptyInput = ")";
        InputStream stream = new ByteArrayInputStream(emptyInput.getBytes());
        var parser = new Parser(stream);

        Tree A = parser.parseA();
        Tree.resetId();

        assertEquals(
                new Tree("A",
                        new Tree("ε")), A);
    }


    /**
     * Тест разбора аргументов функции. Правило {@code A -> T NB}.
     */
    public void testSingleArgumentParseA() throws ParseException {
        String singleArgument = "int x)";
        InputStream stream = new ByteArrayInputStream(singleArgument.getBytes());
        var parser = new Parser(stream);

        Tree A = parser.parseA();
        Tree.resetId();

        assertEquals(
                new Tree("A",
                        new Tree("T",
                                new Tree("N",
                                        new Tree("int")),
                                new Tree("P",
                                        new Tree("ε"))),
                        new Tree("N",
                                new Tree("x")),
                        new Tree("B",
                                new Tree("ε"))), A);
    }

    /**
     * Тест разбора аргументов функции. Правило {@code A -> T NB}.
     */
    public void testTwoArgumentParseA() throws ParseException {
        String singleArgument = "int x, float * y)";
        InputStream stream = new ByteArrayInputStream(singleArgument.getBytes());
        var parser = new Parser(stream);

        Tree A = parser.parseA();
        Tree.resetId();

        assertEquals(
                new Tree("A",
                        new Tree("T",
                                new Tree("N",
                                        new Tree("int")),
                                new Tree("P",
                                        new Tree("ε"))),
                        new Tree("N", new Tree("x")),
                        new Tree("B",
                                new Tree(","),
                                new Tree("T",
                                        new Tree("N",
                                                new Tree("float")),
                                        new Tree("P",
                                                new Tree("*"),
                                                new Tree("P",
                                                        new Tree("ε")))),
                                new Tree("N",
                                        new Tree("y")),
                                new Tree("B",
                                        new Tree("ε")))), A);
    }

    /**
     * Тест разбора аргументов функции, следующих за первым. Правило {@code B -> ε}.
     */
    public void testEmptyInputParseB() throws ParseException {
        String emptyInput = ")";
        InputStream stream = new ByteArrayInputStream(emptyInput.getBytes());
        var parser = new Parser(stream);

        Tree B = parser.parseB();
        Tree.resetId();

        assertEquals(
                new Tree("B",
                        new Tree("ε")), B);
    }

    /**
     * Тест разбора аргументов функции, следующих за первым. Правило {@code B -> , T NB}.
     */
    public void testNonEmptyInputParseB() throws ParseException {
        String emptyInput = ", int value)";
        InputStream stream = new ByteArrayInputStream(emptyInput.getBytes());
        var parser = new Parser(stream);

        Tree B = parser.parseB();
        Tree.resetId();

        assertEquals(
                new Tree("B",
                        new Tree(","),
                        new Tree("T",
                                new Tree("N",
                                        new Tree("int")),
                                new Tree("P",
                                        new Tree("ε"))),
                        new Tree("N",
                                new Tree("value")),
                        new Tree("B",
                                new Tree("ε"))), B);
    }

    /**
     * Тест разбора идентификаторов. Правило {@code N -> w}.
     */
    public void testWithJustWordParseN() throws ParseException {
        String justWord = "justWord";
        InputStream stream = new ByteArrayInputStream(justWord.getBytes());
        var parser = new Parser(stream);

        Tree N = parser.parseN();
        Tree.resetId();

        assertEquals(
                new Tree("N",
                        new Tree(justWord)), N);
    }
}