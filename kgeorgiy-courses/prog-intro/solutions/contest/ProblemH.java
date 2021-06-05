import java.io.*;
import java.nio.charset.Charset;
import java.util.Arrays;
import java.util.InputMismatchException;

public class ProblemH {

    public static void main(String[] args) {
        Scanner in = new Scanner(System.in);

        int n = in.nextIntInLine();
        in.goToNextLine();

        int max = 0;
        int A = 0;

        int[] a = new int[n];
        for (int i = 0; i < n; i++) {
            int current = in.nextIntInLine();
            a[i] = current;
            A += current;
            if (current > max) {
                max = current;
            }
        }
        in.goToNextLine();

        int[] f = new int[A];
        int j = 0;
        for (int i = 0; i < n; i++) {
            for (int k = 0; k < a[i]; k++) {
                f[j++] = i;
            }
        }

        for (int i = 1; i < n; i++) {
            a[i] += a[i - 1];
        }

        int q = in.nextIntInLine();
        in.goToNextLine();

        int[] answer = new int[A + 1];
        Arrays.fill(answer, -1);

        for (int i = 0; i < q; i++) {
            int t = in.nextIntInLine();

            if (t < max) {
                System.out.println("Impossible");
                continue;
            }

            if (answer[t] < 0) {
                int count = 0;
                int b = 0;
                while (b + t < A) {
                    int transaction = f[b + t];
                    b = a[transaction - 1];
                    count++;
                }
                answer[t] = count + 1;
            }

            System.out.println(answer[t]);
        }
        in.close();


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

        public interface Checker {
            boolean isSuitable(int codePoint);
        }
    }
}