package org.cxx.highlighter;

import org.antlr.v4.runtime.ParserRuleContext;
import org.antlr.v4.runtime.tree.TerminalNode;

import java.util.List;

public class LanguageListener extends languageBaseListener {

    final StringBuilder result;
    final Theme theme;
    int styleDepth = 0;
    int currentLine = 1;

    public LanguageListener(Theme theme) {
        this.result = new StringBuilder();
        this.theme = theme;
    }

    private String escapeForHtml(final String string) {
        return string
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace(" ", "&nbsp;");
    }

    private void print(final String string) {
        result.append(string);
    }

    private void printf(final String format, final Object... args) {
        print(String.format(format, args));
    }

    private void addLineNumber() {
        setStyle("numeration");
        print(escapeForHtml(String.format("%4d ", currentLine++)));
        resetStyle();
        print("&nbsp;");
    }

    @Override
    public void enterMain(languageParser.MainContext ctx) {
        printf("<html>%n%n");

        printf("<head>%n");
        printf("    <title>C++ Highlighter</title>%n");
        printf("    <style>%n");
        final List<String> imports = theme.imports();
        if (imports != null) {
            for (final String url : imports) {
                printf("        @import url(%s);%n%n", url);
            }
        }
        printf("        body {%s}%n", theme.commonStyle());
        printf("        span.keyword {%s}%n", theme.keywordsStyle());
        printf("        span.string {%s}%n", theme.stringsStyle());
        printf("        span.primitive {%s}%n", theme.literalsStyle());
        printf("        span.member {%s}%n", theme.membersStyle());
        printf("        span.function {%s}%n", theme.functionDefinitionStyle());
        printf("        span.operator {%s}%n", theme.operatorsStyle());
        printf("        span.numeration {%s}%n", theme.numerationStyle());
        printf("        span.macro {%s}%n", theme.macroStyle());
        printf("        span.constant {%s}%n", theme.constantStyle());
        printf("    </style>%n%n");
        printf("</head>%n%n");

        printf("<body>%n");
        addLineNumber();
    }

    @Override
    public void exitMain(languageParser.MainContext ctx) {
        printf("</body>%n%n</html>%n");
    }

    private void setStyle(final String style) {
        printf("<span class=\"%s\">", style);
        styleDepth += 1;
    }

    private void resetStyle() {
        if (styleDepth == 0) {
            return;
        }
        print("</span>");
        styleDepth -= 1;
    }

    @Override
    public void enterKeyword(languageParser.KeywordContext ctx) {
        setStyle("keyword");
    }

    @Override
    public void enterPrimitive(languageParser.PrimitiveContext ctx) {
        setStyle("primitive");
    }

    @Override
    public void enterString(languageParser.StringContext ctx) {
        setStyle("string");
    }

    @Override
    public void enterChar(languageParser.CharContext ctx) {
        setStyle("string");
    }

    @Override
    public void enterMember(languageParser.MemberContext ctx) {
        setStyle("member");
    }


    @Override
    public void enterUppercase(languageParser.UppercaseContext ctx) {
        setStyle("constant");
    }

    @Override
    public void enterFunctionName(languageParser.FunctionNameContext ctx) {
        setStyle("function");
    }

    @Override
    public void enterIncludePath(languageParser.IncludePathContext ctx) {
        setStyle("string");
    }

    @Override
    public void enterIncludeDirective(languageParser.IncludeDirectiveContext ctx) {
        setStyle("macro");
    }

    @Override
    public void enterOperator(languageParser.OperatorContext ctx) {
        setStyle("operator");
    }

    @Override
    public void exitEveryRule(ParserRuleContext ctx) {
        resetStyle();
    }

    @Override
    public void visitTerminal(TerminalNode node) {
        String style = null;
        switch (node.getSymbol().getType()) {
            case languageParser.EOF -> {
                return;
            }
            case languageParser.NewLine -> {
                print("<br>\n");
                addLineNumber();
                return;
            }
            case languageParser.Bool -> style = "keyword";
            case languageParser.OpeningBracket,
                    languageParser.CodeBlockClosingBracket,
                    languageParser.ClosingBracket,
                    languageParser.OpeningSquareBracket,
                    languageParser.ClosingSquareBracket,
                    languageParser.CodeBlockOpeningBracket,
                    languageParser.Semicolon -> style = "operator";
        }
        if (style != null) {
            setStyle(style);
            print(escapeForHtml(node.getText()));
            resetStyle();
        }
        else {
            print(escapeForHtml(node.getText()));
        }
    }

    public String getResult() {
        return result.toString();
    }
}
