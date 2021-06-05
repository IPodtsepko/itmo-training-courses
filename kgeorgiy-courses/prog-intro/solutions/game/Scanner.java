package game;

import java.io.*;
import java.nio.charset.Charset;
import java.util.InputMismatchException;

public class Scanner {

    private final Reader in;

    public interface Checker {
        boolean isSuitable(int codePoint);
    }
    private Checker checker = codePoint -> !Character.isWhitespace(codePoint);

    private String token = "";

    private boolean EOL = false;
    private boolean EOF = false;

    public Scanner(InputStream in) {
        this.in = new BufferedReader(new InputStreamReader(in));
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
                    } else if (isEOL(codePoint)) {
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

    public boolean isEOL(int codePoint) throws IOException {
        if ((char) codePoint == '\n') {
            return true;
        } else if ((char) codePoint == '\r') {
            in.mark(1);
            if ((char) in.read() != '\n') {
                in.reset();
            }
            return true;
        }
        return false;
    }

    public boolean hasNextInLine() {
        if (EOL) { return false; }
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
        token = "";
        if (EOL) {
            EOL = false;
            return;
        }
        try {
            while (true) {
                int codePoint = in.read();
                if (isEOL(codePoint)) {
                    EOL = false;
                    return;
                } else if (codePoint == -1) {
                    EOF = true;
                    return;
                }
            }
        } catch (IOException e) {
            System.err.println(e.getMessage());
            close();
        }
    }

    public void close() {
        try {
            in.close();
        } catch (IOException e) {
            System.err.println(e.getMessage());
        }
    }
}
