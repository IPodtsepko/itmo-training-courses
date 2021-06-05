import java.util.ArrayList;
import java.util.Arrays;
import java.util.Scanner;

public class ProblemD {

    public static void main(String[] args) {
        final int MOD = 998_244_353;

        Scanner in = new Scanner(System.in);
        int n = in.nextInt();
        int k = in.nextInt();

        IntList[] divisors = new IntList[n + 1];

        for (int i = 0; i < n + 1; i++) {
            divisors[i] = new IntList();
        }

        for (int i = 1; i < n + 1; i++) {
            for (int j = i; j < n + 1; j += i) {
                divisors[j].add(i);
            }
        }

        long[] D = new long[n + 1];
        long[] powers = new long[n + 1];
        powers[0] = 1;
        for (int i = 1; i < n + 1; i++) {
            powers[i] = (powers[i - 1] * k) % MOD;

            Long R;
            if (i % 2 > 0) {
                R = (powers[(i + 1) / 2] * i) % MOD;
            } else {
                R = (((powers[i/2] + powers[i/2 + 1]) % MOD) * i/2) % MOD;
            }

            D[i] = R;
            for (int j = 0; j < divisors[i].length(); j++) {
                int l = divisors[i].get(j);
                if (l < i) {
                    D[i] = (D[i] - (i/l * D[l]) % MOD + MOD) % MOD;
                }
            }
        }

        long answer = 0;
        for (int i = 1; i < n + 1; i++) {
            for (int j = 0; j < divisors[i].length(); j++) {
                int l = divisors[i].get(j);
                answer = (answer + D[l]) % MOD;
            }
        }

        System.out.println(answer);
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
    }
}