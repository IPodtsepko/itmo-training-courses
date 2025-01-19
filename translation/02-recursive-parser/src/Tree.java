import java.io.*;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Objects;


/**
 * @author Игорь Подцепко (i.podtsepko2002@gmail.com
 */
public class Tree {
    final String name;
    final List<Tree> children;
    static int id = 0;

    public Tree(final String name, final Tree... children) {
        this(name, Arrays.asList(children));
    }

    public static void resetId() {
        id = 0;
    }

    public Tree(final String name) {
        this(name, Collections.emptyList());
    }

    public Tree(final String name, final List<Tree> children) {
        this.name = String.format("[%d] %s", ++id, name);
        this.children = children.stream().filter(Objects::nonNull).toList();
    }

    public void exportTo(final Writer writer) throws IOException {
        writer.write(String.format("digraph Tree {%n"));
        exportEdgesTo(writer);
        writer.write(String.format("}%n"));
    }

    private void exportEdgesTo(final Writer writer) throws IOException {
        for (final var child : children) {
            writer.write(String.format("    \"%s\" -> \"%s\"%n", name, child.name));
        }
        for (final var child : children) {
            child.exportEdgesTo(writer);
        }
    }

    @Override
    public boolean equals(Object obj) {
        if (!(obj instanceof Tree tree)) {
            return false;
        }
        if (!Objects.equals(name, tree.name) || children.size() != tree.children.size()) {
            return false;
        }
        for (int i = 0; i < children.size(); ++i) {
            if (!Objects.equals(children.get(i), tree.children.get(i))) {
                return false;
            }
        }
        return true;
    }
}
