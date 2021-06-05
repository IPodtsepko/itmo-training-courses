import java.io.*;
import java.nio.charset.Charset;
import java.util.Arrays;
import java.util.InputMismatchException;

public class ProblemE {

    private static IntList[] adjacencyList;
    private static int[] previous;
    private static int[] distances;

    public static void main(String[] args) {
        Scanner in = new Scanner(System.in);

        int n = in.nextIntInLine();
        int m = in.nextIntInLine();
        in.goToNextLine();

        adjacencyList = new IntList[n];
        for (int i = 0; i < n; i++) {
            adjacencyList[i] = new IntList();
        }

        previous = new int[n];
        distances = new int[n];

        for (int i = 0; i < n - 1; i++) {
            int v = in.nextIntInLine() - 1;
            int u = in.nextIntInLine() - 1;
            in.goToNextLine();

            adjacencyList[v].add(u);
            adjacencyList[u].add(v);
        }

        int[] citiesOfTeam = new int[m];
        for (int i = 0; i < m; i++) {
            citiesOfTeam[i] = in.nextIntInLine() - 1;
        }
        in.close();

        updateDistances(citiesOfTeam[0]);

        int deepest = citiesOfTeam[0];
        int maxDistance = 0;
        for (int city : citiesOfTeam) {
            if (distances[city] > maxDistance) {
                maxDistance = distances[city];
                deepest = city;
            }
        }

        if (maxDistance % 2 != 0) {
            System.out.println("NO");
            return;
        }

        int d = maxDistance / 2;

        int middle = deepest;
        for (int i = 0; i < d; i++) {
            middle = previous[middle];
        }

        updateDistances(middle);

        int distance = distances[citiesOfTeam[0]];
        for (int i = 1; i < m; i++) {
            if (distances[citiesOfTeam[i]] != distance) {
                    System.out.println("NO");
                    return;
            }
        }

        System.out.println("YES");
        System.out.println(middle + 1);
    }

    private static void updateDistances(int start) {
        int n = adjacencyList.length;
        for (int i = 0; i < n; i++) {
            distances[i] = -1;
        }
        updateDistances(start, 0);
    }

    private static void updateDistances(int current, int distance) {
        distances[current] = distance;
        for (int child : adjacencyList[current].getArray()) {
            if (distances[child] < 0) {
                previous[child] = current;
                updateDistances(child, distance + 1);
            }
        }
    }

    public static interface Checker {
        boolean isSuitable(int codePoint);
    }

    public static class IntList {

        private int[] array = new int[1];
        private int length = 0;

        public void add(int value) {
            if (length == array.length) {
                array = Arrays.copyOf(array, length * 2);
            }
            array[length++] = value;
        }

        public int get(int i) { return array[i]; }

        public int length() {
            return length;
        }

        public int[] getArray() {
            return Arrays.copyOf(array, length);
        }
    }

    public static class Scanner {

        private final Reader in;


        private Checker checker = codePoint -> !Character.isWhitespace(codePoint);

        private String token = "";

        private boolean EOL = false;
        private boolean EOF = false;

        public Scanner(InputStream in) {
            this.in = new BufferedReader(new InputStreamReader(in));
        }

        public Scanner(InputStream in, Checker checker) {
            this.in = new BufferedReader(new InputStreamReader(in));
            this.checker = checker;
        }

        public Scanner(String fileName, String charsetName) throws IOException {
            in = new BufferedReader(new FileReader(fileName, Charset.forName(charsetName)));
        }

        public Scanner(String fileName, String charsetName, Checker checker) throws IOException {
            in = new BufferedReader(new FileReader(fileName, Charset.forName(charsetName)));
            this.checker = checker;
        }

        private String readToken() {
            if (token.isEmpty() && !EOL && !EOF) {
                StringBuilder tokenBuilder = new StringBuilder();
                boolean isToken = false;
                try {
                    while (!EOF && !EOL) {
                        int codePoint = in.read();
                        boolean isTokenChar = checker.isSuitable(codePoint);
                        if (codePoint == -1) {
                            EOF = true;
                            close();
                        } else if ((char) codePoint == '\n') {
                            EOL = true;
                        } else if ((char) codePoint == '\r') {
                            in.mark(1);
                            if ((char) in.read() != '\n') {
                                in.reset();
                            }
                            EOL = true;
                        } else if (isTokenChar) {
                            tokenBuilder.append((char) codePoint);
                            isToken = true;
                        } else if (isToken) {
                            break;
                        }
                    }
                    return tokenBuilder.toString();
                } catch (IOException e) {
                    System.err.println(e.getMessage());
                    close();
                }
            }
            return token;
        }

        public boolean hasNextInLine() {
            token = readToken();
            return !token.isEmpty();
        }

        public boolean hasNextIntInLine() {
            token = readToken();
            try {
                if (token.toLowerCase().startsWith("0x")) {
                    Integer.parseUnsignedInt(token.substring(2), 16);
                } else {
                    Integer.parseInt(token);
                }
                return true;
            } catch (NumberFormatException e) {
                return false;
            }
        }

        public boolean hasNextLine() {
            return !EOF;
        }

        public String nextInLine() {
            token = readToken();
            if (!hasNextInLine()) {
                if (EOF) {
                    throw new InputMismatchException("reading after EOF");
                } else if (EOL) {
                    throw new InputMismatchException("reading in ended line");
                }
            }
            String next = token;
            token = "";
            return next;
        }

        public int nextIntInLine() throws NumberFormatException {
            token = readToken();
            int nextInt;
            if (token.toLowerCase().startsWith("0x")) {
                nextInt = Integer.parseUnsignedInt(token.substring(2), 16);
            } else {
                nextInt = Integer.parseInt(token);
            }
            token = "";
            return nextInt;
        }

        public void goToNextLine() {
            EOL = false;
        }

        public void close() {
            try {
                in.close();
            } catch (IOException e) {
                System.err.println(e.getMessage());
            }
        }
    }

}
