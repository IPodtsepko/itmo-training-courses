package org.generator;

import picocli.CommandLine;
import picocli.CommandLine.*;

import org.antlr.v4.runtime.CharStream;
import org.antlr.v4.runtime.CharStreams;
import org.antlr.v4.runtime.CommonTokenStream;
import org.generator.util.GrammarItem;
import org.generator.util.Import;
import org.generator.util.LexerRule;
import org.generator.util.ParserRule;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.stream.Collectors;

@SuppressWarnings("ResultOfMethodCallIgnored")
public class TranslatorGenerator {
    @Command(name = "TranslatorGenerator",
            description = "generates a translator based on a context-free grammar of type LL(1)")
    static class Application implements Callable<Integer> {
        @Option(names = {"-g", "--grammar"}, required = true, description = "path to the file with grammar description")
        Path grammar;

        @Option(names = {"-p", "--project-path"},
                required = true,
                description = "path to the project folder; files will be generated to the specified folder")
        String projectFolder;

        @Option(names = {"-n", "--package"},
                required = true,
                description = "the name of the package to be specified for these files")
        String packageName;

        @SuppressWarnings("FieldMayBeFinal")
        @Option(names = {"-h", "--help"},
                usageHelp = true,
                description = "display a help message")
        private boolean helpRequested = false;

        @Override
        public Integer call() throws Exception {
            TranslatorGenerator.generateGrammar(grammar, projectFolder, packageName);
            return 0;
        }
    }

    public static void main(String[] args) {
        new CommandLine(new Application()).execute(args);
    }

    public static void generateGrammar(final Path grammar,
                                       final String projectFolder,
                                       final String packageName) throws IOException {
        final String input = Files.readAllLines(grammar).stream().collect(Collectors.joining(System.lineSeparator()));
        final CharStream charStream = CharStreams.fromString(input);

        final GrammarLexer lexer = new GrammarLexer(charStream);

        final CommonTokenStream tokenStream = new CommonTokenStream(lexer);
        final GrammarParser parser = new GrammarParser(tokenStream);

        final GrammarParser.GrammarDescriptionContext grammarDescriptionContext = parser.grammarDescription();
        final List<GrammarItem> grammarItems = grammarDescriptionContext.values;

        generateTranslator(projectFolder, packageName, grammarItems);
    }

    public static void generateTranslator(final String projectFolder, final String packageName, List<GrammarItem> items) {
        List<Import> imports = items.stream()
                .filter(item -> item instanceof Import)
                .map(item -> (Import) item)
                .collect(Collectors.toList());
        List<ParserRule> parserRules = items.stream()
                .filter(item -> item instanceof ParserRule)
                .map(item -> (ParserRule) item)
                .collect(Collectors.toList());
        // Parser:
        final String parserCode = ParserGenerator.generate(packageName, imports, parserRules);

        List<LexerRule> lexerRules = items.stream()
                .filter(item -> item instanceof LexerRule)
                .map(item -> (LexerRule) item)
                .collect(Collectors.toList());
        // Enumeration for all terminals:
        final String tokenCode = TokenGenerator.generate(packageName, lexerRules);
        // Lexer:
        final String lexerCode = LexerGenerator.generate(packageName, lexerRules);

        final File packageFolder = new File(projectFolder, packageName);
        packageFolder.mkdir();

        final File parser = new File(packageFolder, "Parser.java");
        write(parser, parserCode);

        final File token = new File(packageFolder, "Token.java");
        write(token, tokenCode);

        final File lexer = new File(packageFolder, "Lexer.java");
        write(lexer, lexerCode);

    }

    private static void write(final File file, final String code) {
        try {
            file.createNewFile();
            try (final FileWriter writer = new FileWriter(file)) {
                writer.write(code);
            }
        } catch (final IOException exception) {
            System.err.println(exception.getLocalizedMessage());
        }
    }
}
