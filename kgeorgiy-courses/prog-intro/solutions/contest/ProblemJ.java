import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.sql.SQLOutput;

public class ProblemJ {
    public static void main(String[] args) {
        int[][] adjacencyMatrix;
        int n;
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(System.in))) {
            n = Integer.parseInt(reader.readLine());
            adjacencyMatrix = new int[n][n];
            for (int i = 0; i < n; i++) {
                String line = reader.readLine();
                for (int j = 0; j < n; j++) {
                    adjacencyMatrix[i][j] = Integer.parseInt("" + line.charAt(j));
                }
            }
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        for (int begin = 0; begin < n; begin++) {
            for (int end = begin + 1; end < n; end++) {
                if (adjacencyMatrix[begin][end] > 0) {
                    for (int i = end + 1; i < n; i++) {
                        adjacencyMatrix[begin][i] = (adjacencyMatrix[begin][i] - adjacencyMatrix[end][i] + 10) % 10;
                    }
                }
            }
        }

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                System.out.print(adjacencyMatrix[i][j]);
            }
            System.out.println();
        }
    }
}
