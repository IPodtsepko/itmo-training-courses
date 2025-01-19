import org.junit.jupiter.api.Test;

import headers.Parser;
import org.generator.util.Tree;

import java.text.ParseException;

public class HeadersTest {
    private static int index = 0;
    
    @Test
    void parseAndShowTree() throws ParseException {
        final Parser parser = new Parser("std::string to_string(boost::uuid * purchase_uuid, Purchase * purchase)");
        final Parser.FunctionHeaderContext ctx = parser.functionHeader();

        final StringBuilder sb = new StringBuilder("digraph {").append(System.lineSeparator());
        pushTree(ctx, -1, 0, sb);
        sb.append("}");

        System.out.println(sb);
    }

    private static void pushTree(final Tree tree, final int id, final int currentId, final StringBuilder stringBuilder) {
        stringBuilder.append(String.format("    %d [label = \"%s\"]%n", currentId, tree.name));
        if (tree.children.isEmpty()) {
            stringBuilder.append(String.format("    %d -> %d%n", id, currentId));
            index += 1;
            return;
        }
        if (id != -1) {
            stringBuilder.append(String.format("    %d -> %d%n", id, currentId));
        }
        for (final Tree child : tree.children) {
            pushTree(child, currentId, ++index, stringBuilder);
        }
    }
}
