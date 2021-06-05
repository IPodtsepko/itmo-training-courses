import java.util.Scanner;

public class ProblemB {
    public static void main(String[] args) {
        Scanner in = new Scanner(System.in);
        int n = in.nextInt();

        int a = -710 * 25_000;
        for (int i = 0; i < n; i++) {
            System.out.println(a);
            a += 710;
        }
    }
}
