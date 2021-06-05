package search;

/**
 * Title task: "Homework 1. Binary search. Hoare triple ("Span" modification)"
 * @author Igor Podtsepko (i.podtsepko@outlook.com)
 */
public class BinarySearchSpan {
    // pre (valid data):
    //      args.length > 0 &&
    //      for all i in [0 : args.length) args[i] contains int &&
    //      for all i in [1 : args.length - 1) a[i] >= a[i + 1]
    public static void main(final String[] args) {
        // pre: valid data
        final int n = args.length - 1;
        // post: n == args.length - 1 && valid data

        // pre: valid data && n == args.length - 1
        final int x = Integer.parseInt(args[0]);
        // post: x == args[0] (as integer) && valid data && n == args.length - 1

        // pre: x == args[0] (as integer) && valid data && n == args.length - 1
        final int[] a = new int[n];
        // post: x == args[0] (as integer) && valid data && n == args.length - 1 &&
        //       a.length == n &&
        //       a isn't null

        // pre: x == args[0] (as integer) && valid data && n == args.length - 1 &&
        //      a.length = n &&
        //      a isn't null
        for (int i = 0; i < n; i++) {
            a[i] = Integer.parseInt(args[i + 1]);
        }
        // post: x == args[0] (as integer) && valid data && n == args.length - 1 &&
        //       for all i in [0 : n) a[i] == args[i + 1] (as integer) --> valid a:
        //               a isn't null && for all i in [0 : n - 1) a[i] >= a[i + 1]

        // pre: valid a
        final int left = iterativeBinarySearch(x, a);
        // post: valid a && left found: left == min({i : i in [0 : n) && a[i] <= x} ∪ {n})

        // pre: valid a && left found
        final int right = x > Integer.MIN_VALUE ? recursiveBinarySearch(x - 1, a) : n;
        // post (range found):
        //          left found &&
        //          for all i in [left : right) a[i] <= x &&
        //          for all i in [right : n) a[i] < x

        // pre: for all i in [left : right) a[i] <= x && for all i in [right : n) a[i] < x
        final int length = right == 0 ? 0 : right - left;
        // post: range found && length = [left : right).length

        System.out.println(left + " " + length);
    }

    // pre (valid a):
    //     a isn't null && for all i in [0 : n - 1) a[i] >= a[i + 1]
    public static int iterativeBinarySearch(final int x, final int[] a) {
        // pre: valid a
        int left = -1;
        // post: valid a && left == -1

        // pre: valid a && left == -1
        int right = a.length;
        // post: valid a && left == -1 < right == n

        // Inv: for all i in [0 : left]       a[i] > x  &&
        //      for all i in [right : n - 1]  a[i] <= x &&
        //      -1 <= left < right <= n &&
        //      valid a
        while (left + 1 != right) {
            // pre: Inv
            final int middle = right + (left - right) / 2;
            // post: middle == [(left + right) / 2] (integer part) &&
            //       left + 1 < right ---> left < right - 1 &&
            //
            //       middle <= (left + right) / 2 < ((right - 1) + right) / 2 = right - 1 / 2 < right
            //       (middle < right) &&
            //
            //       middle > (left + right) / 2 - 1 >= (left + (left + 2)) / 2 - 1 > (left + 1) - 1 = left
            //       (middle > left) &&
            //
            //       |---> left < middle < right && middle in [0 : n) && Inv

            // pre: left < middle < right && middle in [0 : n) && Inv
            final int current = a[middle];
            // post: left < middle < right && middle in [0 : n) && Inv && current == a[middle]

            // pre: left < middle < right && middle in [0 : n) && Inv && current == a[middle]
            if (current > x) {
                left = middle;
                // post: left increased && Inv
            } else {
                right = middle;
                // post: right decreased && Inv
            }
            // post: Inv && (right - left) decreased --> the cycle will end
        }
        // post: Inv && left + 1 == right == min({i : i in [0 : n) && a[i] <= x} ∪ {n})

        return right;
    }
    // post: R == min({i : i in [0 : n) && a[i] <= x} ∪ {n})

    // pre (Inv): for all i in [0 : left]       a[i] > x  &&
    //            for all i in [right : n - 1]  a[i] <= x &&
    //            -1 <= left < right <= n &&
    //            valid a:
    //                  a isn't null && for all i in [0 : n - 1) a[i] >= a[i + 1]
    public static int recursiveBinarySearch(final int x, final int[] a, final int left, final int right) {
        // pre: Inv
        if (left + 1 == right) {
            // right == min({i : i in [0 : n) && a[i] <= x} ∪ {n})
            return right;
        }
        // post: Inv && left + 1 < right

        // pre: Inv && left + 1 < right
        final int middle = right + (left - right) / 2;
        // post: left < middle == [(left + right) / 2] < right (see iterative algorithm) &&
        //       middle in [0 : n - 1] &&
        //       Inv

        // pre: left < middle == [(left + right) / 2] < right &&
        //      middle in [0 : n - 1] &&
        //      Inv
        final int current = a[middle];
        // post: left < middle == [(left + right) / 2] < right &&
        //       middle in [0 : n - 1] &&
        //       current == a[middle] &&
        //       Inv

        // pre: left < middle == [(left + right) / 2] < right &&
        //       middle in [0 : n - 1] &&
        //       current == a[middle] &&
        //       Inv
        if (current > x) {
            // for all i in [0 : middle]     a[i] > x  &&
            // for all i in [right : n - 1]  a[i] <= x &&
            // right - left > right - middle (segment decreased) --> extreme case will be reached
            return recursiveBinarySearch(x, a, middle, right);
        } else {
            // for all i in [0 : left]       a[i] > x  &&
            // for all i in [middle : n - 1] a[i] <= x &&
            // right - left > middle - left (segment decreased) --> extreme case will be reached
            return recursiveBinarySearch(x, a, left, middle);
        }
    }
    // post: R == min({i : i in (left : right) && a[i] <= x} ∪ {n})


    // pre (valid a):
    //     a isn't null && for all i in [0 : n - 1) a[i] >= a[i + 1]
    public static int recursiveBinarySearch(final int x, final int[] a) {
        return recursiveBinarySearch(x, a, -1, a.length);
    }
    // post: R == min({i : i in [0 : n) && a[i] <= x} ∪ {n})
}