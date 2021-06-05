package md2html;

import md2html.parser.PageParser;
import md2html.parser.exceptions.ParseException;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.StringReader;
import java.util.List;

/**
 * @author Igor Podtsepko (i.podtsepko@niuitmo.com)
 */
public class Main {
    public static final String ANSI_RED = "\u001B[31m";
    public static final String ANSI_GREEN = "\u001B[32m";
    public static final String ANSI_RESET = "\u001B[0m";

    public static void main(String[] args) {
        List<String> mdPages = List.of(
                "*hello", "\\*hello", "**hello", "\\**hello", "\\*\\*hello",
                "*hello _world*", "*hello \\_world*", "**this is ![image](address)",
                "this is ![image](address)", "this is ![image](address",
                "this is ![image(address)", "this is \\![image]", "this is ![image ![trash](address)",
                "this is ![image ![trash](address)](trash)"
                );
        for (String mdPage : mdPages) {
            System.out.print(mdPage + ": ");
            try (BufferedReader input = new BufferedReader(new StringReader(mdPage))) {
                StringBuilder resultPage = new StringBuilder();
                new PageParser().parse(input).toHtml(resultPage);
                System.out.print(ANSI_GREEN + resultPage.toString() + ANSI_RESET);
            } catch (IOException | ParseException e) {
                System.out.println(ANSI_RED + e.getMessage() + ANSI_RESET);
            }
        }
    }
}
