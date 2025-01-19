package org.cxx.highlighter;

import org.antlr.v4.runtime.CharStream;
import org.antlr.v4.runtime.CharStreams;
import org.antlr.v4.runtime.CommonTokenStream;
import org.antlr.v4.runtime.tree.ParseTreeWalker;

import java.io.IOException;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

public class LanguageHighlighter {
    public static String highlight(String program, Theme theme) {
        CharStream charStream = CharStreams.fromString(program);
        languageLexer lexer = new languageLexer(charStream);
        CommonTokenStream tokens = new CommonTokenStream(lexer);
        languageParser parser = new languageParser(tokens);
        languageParser.MainContext context = parser.main();

        ParseTreeWalker walker = new ParseTreeWalker();
        LanguageListener listener = new LanguageListener(theme);

        walker.walk(listener, context);

        return listener.getResult();
    }

    public static void main(String[] args) throws IOException {
        Theme oneDark = new Theme() {
            @Override
            public List<String> imports() {
                return List.of("https://fonts.googleapis.com/css2?family=JetBrains+Mono:ital@0;1&display=swap");
            }

            @Override
            public String commonStyle() {
                return new Style()
                        .setFontFamily("'JetBrains Mono', monospace")
                        .setFontSize("14px")
                        .setColor("rgb(220, 223, 228)")
                        .setBackgroundColor("rgb(40, 44, 52)")
                        .build();
            }

            @Override
            public String keywordsStyle() {
                return new Style()
                        .setColor("rgb(198, 120, 221)")
                        .build();
            }

            @Override
            public String stringsStyle() {
                return new Style()
                        .setColor("rgb(152, 195, 121)")
                        .build();
            }

            @Override
            public String literalsStyle() {
                return new Style()
                        .setColor("rgb(229, 192, 123)")
                        .build();
            }

            @Override
            public String membersStyle() {
                return new Style()
                        .setColor("rgb(86, 182, 194)")
                        .setFontStile("italic")
                        .build();
            }

            @Override
            public String functionDefinitionStyle() {
                return new Style()
                        .setColor("rgb(97, 175, 239)")
                        .build();
            }

            @Override
            public String operatorsStyle() {
                return new Style()
                        .setColor("rgb(86, 182, 194)")
                        .build();
            }

            @Override
            public String numerationStyle() {
                return new Style()
                        .setColor("rgb(117, 117, 117)")
                        .setBackgroundColor("rgb(45, 49, 57)")
                        .build();
            }

            @Override
            public String macroStyle() {
                return new Style()
                        .setColor("rgb(224, 108, 117)")
                        .build();
            }

            @Override
            public String constantStyle() {
                return new Style()
                        .setColor("rgb(224, 108, 117)")
                        .setFontStile("italic")
                        .build();
            }
        };
        Path filePath = Path.of("main.cpp");
        String content = Files.readString(filePath);
        try (PrintWriter out = new PrintWriter("index.html")) {
            out.print(LanguageHighlighter.highlight(content, oneDark));
        }
    }
}
