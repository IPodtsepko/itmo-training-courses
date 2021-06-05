package md2html;

import md2html.parser.PageParser;
import md2html.parser.exceptions.ParseException;

import java.io.*;
import java.nio.charset.StandardCharsets;

/**
 * @author Igor Podtsepko (i.podtsepko@niuitmo.com)
 */
public class Md2Html {
    public static void main(String[] args) {
        if (args.length != 2) {
            throw new IllegalArgumentException("Expected input file name and output file name");
        }

        String inputFileName = args[0];
        String outputFileName = args[1];

        try (BufferedReader input = new BufferedReader(new FileReader(inputFileName, StandardCharsets.UTF_8));
             BufferedWriter output = new BufferedWriter(new FileWriter(outputFileName, StandardCharsets.UTF_8)))
        {
            StringBuilder resultPage = new StringBuilder();
            new PageParser().parse(input).toHtml(resultPage);
            output.write(resultPage.toString());
        } catch (IOException | ParseException e) {
            System.err.println(e.getMessage());
        }
    }
}
