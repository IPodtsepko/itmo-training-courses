import java.util.Arrays;
import java.util.Scanner;

public class ProblemD {
    public static final int MOD = 998_244_353;

    public static void main(String[] args) {

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

        long[] d = new long[n + 1];
        long[] powers = new long[n + 1];
        powers[0] = 1;
        for (int i = 1; i < n + 1; i++) {
            powers[i] = (powers[i - 1] * k) % MOD;

            long r;
            if (i % 2 > 0) {
                r = (powers[(i + 1) / 2] * i) % MOD;
            } else {
                r = (((powers[i / 2] + powers[i / 2 + 1]) % MOD) * i / 2) % MOD;
            }

            d[i] = r;
            for (int j = 0; j < divisors[i].length(); j++) {
                int l = divisors[i].get(j);
                if (l < i) {
                    d[i] = (d[i] - (i / l * d[l]) % MOD + MOD) % MOD;
                }
            }
        }

        long answer = 0;
        for (int i = 1; i < n + 1; i++) {
            for (int j = 0; j < divisors[i].length(); j++) {
                int l = divisors[i].get(j);
                answer = (answer + d[l]) % MOD;
            }
        }

        System.out.println(answer);
    }

    public static class IntList {

        private int[] array = new int[1];
        private int size;

        public void add(int value) {
            if (size == array.length) {
                array = Arrays.copyOf(array, size * 2);
            }
            array[size++] = value;
        }

        public int get(int i) { return array[i]; }

        public int length() {
            return size;
        }
    }
}