import java.io.*;
import java.util.Arrays;

public class ProblemK {
    public static void main(String[] args) {

        Parser in = new Parser(System.in);
        int n = in.nextInt();
        int m = in.nextInt();

        int castleAX = -1;
        int castleAY = -1;
        char[][] field = new char[n][m];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                field[i][j] = (char) in.nextChar();
                if (field[i][j] == 'A') {
                    castleAX = i;
                    castleAY = j;
                }
            }
        }

        int upperBoundA;
        int lowerBoundA;
        int leftBoundA;
        int rightBoundA;
        upperBoundA = lowerBoundA = castleAX;
        leftBoundA = rightBoundA = castleAY;
        int areaA = 1;

        int[] upperBounds = new int[m];
        Arrays.fill(upperBounds, -1);

        int[] leftBounds = new int[m];
        int[] rightBounds = new int[m];

        IntStack s = new IntStack();
        for (int lowerBound = 0; lowerBound < n; lowerBound++) {
            for (int i = 0; i < m; i++) {
                if (field[lowerBound][i] != '.' && field[lowerBound][i] != 'A') {
                    upperBounds[i] = lowerBound;
                }
            }

            s.clear();
            for (int i = 0; i < m; i++) {
                while (!s.empty() && upperBounds[s.top()] <= upperBounds[i]) {
                    s.pop();
                }
                leftBounds[i] = s.empty() ? -1 : s.top();
                s.push(i);
            }

            s.clear();
            for (int i = m - 1; i >= 0; i--) {
                while (!s.empty() && upperBounds[s.top()] <= upperBounds[i]) {
                    s.pop();
                }
                rightBounds[i] = s.empty() ? m : s.top();
                s.push(i);
            }

            for (int i = 0; i < m; i++) {
                if (upperBounds[i] < castleAX && castleAX <= lowerBound &&
                        leftBounds[i] < castleAY && castleAY < rightBounds[i]) {
                    int area = getArea(
                            upperBounds[i] + 1, leftBounds[i] + 1,
                            lowerBound, rightBounds[i] - 1
                    );
                    if (area > areaA) {
                        upperBoundA = upperBounds[i] + 1;
                        leftBoundA = leftBounds[i] + 1;
                        rightBoundA = rightBounds[i] - 1;
                        lowerBoundA = lowerBound;
                        areaA = area;
                    }
                }
            }
        }

        fill(field, upperBoundA, leftBoundA, lowerBoundA, rightBoundA);
        fill(field, 0, 0, upperBoundA - 1, m - 1);
        fill(field, upperBoundA, 0, lowerBoundA, leftBoundA - 1);
        fill(field, upperBoundA, rightBoundA + 1, lowerBoundA, m - 1);
        fill(field, lowerBoundA + 1, 0, n - 1, m - 1);

        try (BufferedWriter out = new BufferedWriter(new OutputStreamWriter(System.out))) {
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < m; j++) {
                    out.write(field[i][j]);
                }
                out.newLine();
            }
        } catch (IOException e) {
            new RuntimeException(e);
        }

    }

    public static int getArea(int x1, int y1, int x2, int y2) {
        if (x1 > x2) {
            int x = x1;
            x1 = x2;
            x2 = x;
        }
        if (y1 > y2) {
            int y = y1;
            y1 = y2;
            y2 = y;
        }
        return (x2 + 1 - x1) * (y2 + 1 - y1);
    }

    public static void fill(char[][] field, int upperBound, int leftBound, int lowerBound, int rightBound) {
        for (int i = upperBound; i < lowerBound + 1; i++) {
            for (int j = leftBound; j < rightBound + 1; j++) {
                if (field[i][j] != '.') {
                    char owner = Character.toLowerCase(field[i][j]);
                    int x = i;
                    while (x < lowerBound && field[x + 1][j] == '.') {
                        field[++x][j] = owner;
                    }
                    x = i;
                    while (upperBound < x && field[x - 1][j] == '.') {
                        field[--x][j] = owner;
                    }
                }
            }
        }

        for (int i = upperBound; i < lowerBound + 1; i++) {
            for (int j = leftBound; j < rightBound + 1; j++) {
                if (field[i][j] != '.') {
                    char owner = Character.toLowerCase(field[i][j]);
                    int y = j;
                    while (y < rightBound && field[i][y + 1] == '.') {
                        field[i][++y] = owner;
                    }
                    y = j;
                    while (leftBound < y && field[i][y - 1] == '.') {
                        field[i][--y] = owner;
                    }
                }
            }
        }
    }
}

class Parser {
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
        StringBuilder sb = new StringBuilder();
        boolean sbEmpty = true;
        while (true) {
            int c = read();
            if (!Character.isWhitespace(c)) {
                if (c == -1) {
                    break;
                }
                sb.append((char) c);
                sbEmpty = false;
            } else if (!sbEmpty) {
                break;
            }
        }
        return sb.toString();
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

class IntList {
    private int[] array = new int[1];
    private int length = 0;

    public void add(int value) {
        if (length == array.length) {
            array = Arrays.copyOf(array, length * 2);
        }
        array[length++] = value;
    }

    public void pop() {
        if (length == 0) {
            return;
        }
        length--;
        if (2 * length == array.length) {
            array = Arrays.copyOf(array, length);
        }
    }

    public int get(int i) {
        return array[i];
    }

    public int length() {
        return length;
    }
}

class IntStack {
    private IntList items = new IntList();

    public void push(int value) {
        items.add(value);
    }

    public int pop() {
        int top = items.get(items.length() - 1);
        items.pop();
        return top;
    }

    public int top() {
        return items.get(items.length() - 1);
    }

    public boolean empty() {
        return items.length() == 0;
    }

    public void clear() {
        items = new IntList();
    }
}
