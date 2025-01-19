import junit.framework.TestCase;

import java.io.ByteArrayInputStream;
import java.io.InputStream;
import java.text.ParseException;

public class LexicalAnalyzerTest extends TestCase {

    public void testGetWord() throws ParseException {
        String justWord = "some_word";
        InputStream stream = new ByteArrayInputStream(justWord.getBytes());

        LexicalAnalyzer analyzer = new LexicalAnalyzer(stream);

        assertNull(analyzer.getCurrentToken());
        analyzer.nextToken();

        assertEquals(Token.WORD, analyzer.getCurrentToken());
        assertEquals(justWord, analyzer.getWord());
    }

    public void testGetToken() throws ParseException {
        String justWord = "*(,);";
        InputStream stream = new ByteArrayInputStream(justWord.getBytes());

        LexicalAnalyzer analyzer = new LexicalAnalyzer(stream);
        analyzer.nextToken();

        assertEquals(Token.STAR, analyzer.getCurrentToken());
        analyzer.nextToken();

        assertEquals(Token.OPENING_BRACKET, analyzer.getCurrentToken());
        analyzer.nextToken();

        assertEquals(Token.COMMA, analyzer.getCurrentToken());
        analyzer.nextToken();

        assertEquals(Token.CLOSING_BRACKET, analyzer.getCurrentToken());
        analyzer.nextToken();

        assertEquals(Token.SEMICOLON, analyzer.getCurrentToken());
        analyzer.nextToken();
    }

    public void testTokenizeFullFunctionDeclaration() throws ParseException {
        String justWord = "int ** f(some_type * x, string y, float z);";
        InputStream stream = new ByteArrayInputStream(justWord.getBytes());

        LexicalAnalyzer analyzer = new LexicalAnalyzer(stream);
        analyzer.nextToken();

        assertEquals(Token.WORD, analyzer.getCurrentToken());
        assertEquals("int", analyzer.getWord());
        analyzer.nextToken();

        assertEquals(Token.STAR, analyzer.getCurrentToken());
        analyzer.nextToken();

        assertEquals(Token.STAR, analyzer.getCurrentToken());
        analyzer.nextToken();

        assertEquals(Token.WORD, analyzer.getCurrentToken());
        assertEquals("f", analyzer.getWord());
        analyzer.nextToken();

        assertEquals(Token.OPENING_BRACKET, analyzer.getCurrentToken());
        analyzer.nextToken();

        assertEquals(Token.WORD, analyzer.getCurrentToken());
        assertEquals("some_type", analyzer.getWord());
        analyzer.nextToken();

        assertEquals(Token.STAR, analyzer.getCurrentToken());
        analyzer.nextToken();

        assertEquals(Token.WORD, analyzer.getCurrentToken());
        assertEquals("x", analyzer.getWord());
        analyzer.nextToken();

        assertEquals(Token.COMMA, analyzer.getCurrentToken());
        analyzer.nextToken();

        assertEquals(Token.WORD, analyzer.getCurrentToken());
        assertEquals("string", analyzer.getWord());
        analyzer.nextToken();

        assertEquals(Token.WORD, analyzer.getCurrentToken());
        assertEquals("y", analyzer.getWord());
        analyzer.nextToken();

        assertEquals(Token.COMMA, analyzer.getCurrentToken());
        analyzer.nextToken();

        assertEquals(Token.WORD, analyzer.getCurrentToken());
        assertEquals("float", analyzer.getWord());
        analyzer.nextToken();

        assertEquals(Token.WORD, analyzer.getCurrentToken());
        assertEquals("z", analyzer.getWord());
        analyzer.nextToken();

        assertEquals(Token.CLOSING_BRACKET, analyzer.getCurrentToken());
        analyzer.nextToken();

        assertEquals(Token.SEMICOLON, analyzer.getCurrentToken());
        analyzer.nextToken();
    }
}