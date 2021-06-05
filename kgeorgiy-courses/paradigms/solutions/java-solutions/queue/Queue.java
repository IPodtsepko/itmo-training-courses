package queue;

/**
 * Title task: "Homework 2, 3. Queue: Programming Aproaches, OOP"
 * @author Igor Podtsepko (i.podtsepko@outlook.com)
 */

public interface Queue {

    // pre: element != null
    boolean contains(Object element);
    // post: exist i : a[i] == e

    // pre:
    boolean removeFirstOccurrence(Object element);
    // post: if exist i : for all j < i a[j] != element
    //                    a[i] == element
    //              n = n' - 1 && for all j in [1 : i) a[j] == a'[j] && for all j in [i : n] a[j] = a'[j + 1], R = true
    //       else R = false

    // pre: element != null
    void enqueue(Object element);
    // post: n = n' + 1 && for all i in [1 : n'] a[i] == a'[i] && a[n] = e

    // pre: n > 0
    Object element();
    // post: R = a[1] && Immutable

    // pre: n > 0
    Object dequeue();
    // post: R = a[1] && n = n' - 1 && for all i in [1 : n] a[i] = a'[i + 1]

    // pre: true
    int size();
    // post: R == n && Immutable

    // pre: true
    boolean isEmpty();
    // post: post: R == (n == 0) && Immutable

    // pre: true
    void clear();
    // post: n == 0
}
