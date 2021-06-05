package search;

/**
 * Title task: "Homework 1. Binary search. Hoare triple (base modification)"
 * @author Igor Podtsepko (i.podtsepko@outlook.com)
 */
public class BinarySearch {
    public static void main(String[] args) {
        int n = args.length - 1;
        int x = Integer.parseInt(args[0]);
        int[] a = new int[n];
        for (int i = 0; i < n; i++) {
            a[i] = Integer.parseInt(args[i + 1]);
        }
        System.out.println(iterativeBinarySearch(x, a));
    }

    // pre: true
    public static int iterativeBinarySearch(int x, int[] a) {
        // pre: true
        int left = -1;
        // post: left == -1

        // pre: true
        int right = a.length;
        // post: left == -1, right == n

        // Inv: for all i in [0 : left]       a[i] > x,
        //      for all i in [right : n - 1]  a[i] <= x,
        //      -1 <= left < right <= n,
        while (left + 1 != right) {
            // pre: left < right
            int middle = left + (right - left) / 2;
            // post: middle == [(left + right) / 2] (integer part),
            //       left + 1 < right ---> left < right - 1
            //
            //       middle <= (left + right) / 2 < ((right - 1) + right) / 2 = right - 1 / 2 < right
            //       (middle < right)
            //
            //       middle > (left + right) / 2 - 1 >= (left + (left + 2)) / 2 - 1 > (left + 1) - 1 = left
            //       (middle > left)
            //
            //       |---> middle in [0 : n - 1] and left < middle < right

            // pre: middle in [0 : n - 1]
            int current = a[middle];
            // post: current == a[middle]

            // pre: left < middle < right
            if (current > x) {
                left = middle;
                // post: left increased, Inv
            } else {
                right = middle;
                // post: right decreased, Inv
            }
            // post: Inv, (right - left) decreased ---> the cycle will end
        }

        return right;
    }
    // post: R == n if doesn't exist i : a[i] <= x
    //       else a[R] <= x and if R > 0 then a[R - 1] > x


    // Inv: for all i in [0 : left]       a[i] > x,
    //      for all i in [right : n - 1]  a[i] <= x,
    //      -1 <= left < right <= n
    public static int recursiveBinarySearch(int x, int[] a, int left, int right) {
        // pre: left < right
        if (left + 1 == right) {
            // right == n (index not found) or a[right - 1] <= x and if right > 1 a[right - 2] > x
            return right;
        }
        // post: left + 1 < right

        // pre: left + 1 < right
        int middle = left + (right - left) / 2;
        // post: -1 <= left < middle == (left + right) / 2 < right <= n (see iterative algorithm),
        //       middle in [0 : n - 1]

        // pre: middle in [0 : n - 1]
        int current = a[middle];
        // post: current == a[middle]

        // left < middle < right
        if (current > x) {
            // right - left > right - middle (segment decreased) ---> extreme case will be reached
            return recursiveBinarySearch(x, a, middle, right);
        } else {
            // right - left > middle - left (segment decreased) ---> extreme case will be reached
            return recursiveBinarySearch(x, a, left, middle);
        }
    }
    // post: R == n (index not found) or a[R] <= x and if right > 0 then a[right - 1] > x

    public static int recursiveBinarySearch(int x, int[] a) {
        return recursiveBinarySearch(x, a, -1, a.length);
    }
}