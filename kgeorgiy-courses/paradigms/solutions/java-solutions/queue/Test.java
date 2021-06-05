package queue;

/**
 * Title task: "Homework 3. Queue: OOP"
 * @author Igor Podtsepko (i.podtsepko@outlook.com)
 */

public class Test {
    public static void main(String[] args) {
        LinkedQueue q = new LinkedQueue();
        for (int i = 0; i < 5; i++) {
            q.enqueue(10 * i);
            System.out.println("size = " + q.size());
        }
        System.out.println(q.size() + " (" + q.isEmpty() + ")");
        for (int i = 0; i < 5; i++) {
            System.out.println("q.element() = " + q.element());
            System.out.println("q.dequeue() = " + q.dequeue());
            System.out.println("size = " + q.size());
        }
        System.out.println("size = " + q.size() + " (" + q.isEmpty() + ")");
    }
}
