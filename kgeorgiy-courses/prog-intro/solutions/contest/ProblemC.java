import java.io.*;
import java.util.Arrays;

public class ProblemC {

    private static final int INTERNAL = 0;
    private static final int EXTERNAL = 1;

    public static void main(String[] args) {
        Parser in = new Parser(System.in);

        int w = in.nextInt();
        int h = in.nextInt();

        AdjacencyMatrix graph = new AdjacencyMatrix(w, h);

        int start = -1;

        for (int i = 0; i < h; i++) {
            for (int j = 0; j < w; j++) {
                char c = (char) in.nextChar();
                if (c == 'X') {
                    int current = i * (w + 1) + j;
                    int down = (i + 1) * (w + 1) + j;
                    int right = i * (w + 1) + (j + 1);
                    int diagonal = (i + 1) * (w + 1) + (j + 1);

                    graph.add(INTERNAL, current, down);
                    graph.add(INTERNAL, right, diagonal);
                    graph.add(EXTERNAL, current, diagonal);
                    graph.add(EXTERNAL, down, right);


                    if (start == -1) {
                        start = current;
                    }
                }
            }
        }

        IntList answer = new IntList();
        searchWay(start, INTERNAL, graph, answer);

        try (BufferedWriter out = new BufferedWriter(new OutputStreamWriter(System.out))) {
            out.write(String.format("%d%n", answer.length() - 2));

            for (int i = 0; i < answer.length() - 1; i++) {
                int v = answer.get(i);
                int x = v % (w + 1);
                int y = v / (w + 1);

                out.write(String.format("%d %d%n", x, y));
            }
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    private static void searchWay(int v, int type, AdjacencyMatrix graph, IntList answer) {
        IntList children = graph.get(v, type);
        while (children.length() > 0) {
            int u = children.pop();
            searchWay(u, (type + 1) % 2, graph, answer);
        }
        answer.add(v);
    }

    private static class Parser {
        Reader reader;

        public Parser(InputStream in) {
            reader = new BufferedReader(new InputStreamReader(in));
        }

        public void close() {
            try {
                reader.close();
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        }

        private int read() {
            int c;
            try {
                c = reader.read();
            } catch (IOException e) {
                c = -1;
            }
            if (c < 0) {
                close();
            }
            return c;
        }

        public String next() {
            StringBuilder next = new StringBuilder();
            boolean sbEmpty = true;
            while (true) {
                int c = read();
                if (!Character.isWhitespace(c)) {
                    if (c == -1) {
                        break;
                    }
                    next.append((char) c);
                    sbEmpty = false;
                } else if (!sbEmpty) {
                    break;
                }
            }
            return next.toString();
        }

        public int nextInt() {
            return Integer.parseInt(next());
        }

        public int nextChar() {
            while (true) {
                int c = read();
                if (!Character.isWhitespace(c)) {
                    return c;
                }
            }
        }
    }

    private static class IntList {
        private int[] array = new int[1];
        private int length = 0;

        public void add(int value) {
            if (length == array.length) {
                array = Arrays.copyOf(array, length * 2);
            }
            array[length++] = value;
        }

        public int pop() {
            if (length == 0) {
                throw new RuntimeException("List empty");
            }
            int data = array[--length];
            if (2 * length == array.length) {
                array = Arrays.copyOf(array, length);
            }
            return data;
        }

        public int get(int i) {
            return array[i];
        }

        public int length() {
            return length;
        }
    }

    private static class AdjacencyMatrix {

        private IntList[][] edges;

        public AdjacencyMatrix(int w, int h) {
            edges = new IntList[(h + 1) * (w + 1)][2];
        }

        public void add(int type, int v, int u) {
            if (edges[v][type] == null) {
                edges[v][type] = new IntList();
            }
            if (edges[u][type] == null) {
                edges[u][type] = new IntList();
            }
            edges[v][type].add(u);
            edges[u][type].add(v);
        }

        private IntList get(int v, int type) {
            return edges[v][type];
        }
    }
}