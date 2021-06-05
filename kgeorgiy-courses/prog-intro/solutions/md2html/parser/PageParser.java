package md2html.parser;

import md2html.markup.*;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.util.LinkedList;
import java.util.List;

/**
 * @author Igor Podtsepko (i.podtsepko@niuitmo.com)
 */
public class PageParser {
    private List<BlockFormattedElement> formattedBlocks;
    private StringBuilder block;

    public Page parse(final BufferedReader mdPage) throws IOException {
        formattedBlocks = new LinkedList<>();
        block = new StringBuilder();

        while (true) {
            String line = mdPage.readLine();
            if (line == null) {
                break;
            }
            if (line.isEmpty()) {
                parseBlock();
            } else {
                if (block.length() > 0) {
                    block.append('\n');
                }
                block.append(line);
            }
        }
        parseBlock();
        return new Page(formattedBlocks);
    }

    private void parseBlock() {
        if (block.length() == 0) {
            return;
        }
        int headerLevel = parseHeaderLevel();
        if (headerLevel > 0) {
            formattedBlocks.add(new Header(headerLevel, new InlineMarkupParser().parse(block.substring(headerLevel + 1))));
        } else {
            formattedBlocks.add(new Paragraph(new InlineMarkupParser().parse(block)));
        }
        block = new StringBuilder();
    }

    private int parseHeaderLevel() {
        int level = 0;
        while (level < block.length() && block.charAt(level) == '#') {
            level++;
        }
        if (level < block.length() && block.charAt(level) == ' ') {
            return level;
        }
        return 0;
    }
}