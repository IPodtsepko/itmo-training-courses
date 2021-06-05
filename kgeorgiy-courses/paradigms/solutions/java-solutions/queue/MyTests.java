package queue;

/**
 * Title task: "Homework 2. Queue: Programming Aproaches"
 * @author Igor Podtsepko (i.podtsepko@outlook.com)
 */

public class MyTests {
    public static void main(String[] args) {
        System.out.println("<-- MODULE TESTING -->");
        for (int i = 0; i < 15; i++) {
            ArrayQueueModule.enqueue(i);
            ArrayQueueModule.push(-i);
        }
        System.out.println("Size = " + ArrayQueueModule.size());
        System.out.println("Is empty?\n— " + ArrayQueueModule.isEmpty());
        for (int i = 0; i < 11; i++) {
            System.out.print(ArrayQueueModule.element() + "(" + ArrayQueueModule.peek() + ")" + " ");
            ArrayQueueModule.dequeue();
            ArrayQueueModule.remove();
        }
        ArrayQueueModule.clear();
        System.out.println("Cleansing...");
        System.out.println("Size = " + ArrayQueueModule.size());
        System.out.println("Is empty?\n— " + ArrayQueueModule.isEmpty());

        System.out.println("\n<-- ADT TESTING -->");
        ArrayQueueADT q = new ArrayQueueADT();
        for (int i = 0; i < 15; i++) {
            ArrayQueueADT.enqueue(q, i);
            ArrayQueueADT.push(q, -i);
        }
        System.out.println("Size = " + ArrayQueueADT.size(q));
        System.out.println("Is empty?\n— " + ArrayQueueADT.isEmpty(q));
        for (int i = 0; i < 11; i++) {
            System.out.print(ArrayQueueADT.element(q) + "(" + ArrayQueueADT.peek(q) + ")" + " ");
            ArrayQueueADT.dequeue(q);
            ArrayQueueADT.remove(q);
        }
        ArrayQueueADT.clear(q);
        System.out.println("Cleansing...");
        System.out.println("Size = " + ArrayQueueADT.size(q));
        System.out.println("Is empty?\n— " + ArrayQueueADT.isEmpty(q));

        System.out.println("\n<-- CLASS TESTING -->");
        ArrayQueue arrayQueue = new ArrayQueue();
        for (int i = 0; i < 15; i++) {
            arrayQueue.enqueue(i);
            arrayQueue.push(-i);
        }
        System.out.println("Size = " + arrayQueue.size());
        System.out.println("Is empty?\n— " + arrayQueue.isEmpty());
        for (int i = 0; i < 11; i++) {
            System.out.print(arrayQueue.element() + "(" + arrayQueue.peek() + ")" + " ");
            arrayQueue.dequeue();
            arrayQueue.remove();
        }
        arrayQueue.clear();
        System.out.println("Cleansing...");
        System.out.println("Size = " + arrayQueue.size());
        System.out.println("Is empty?\n— " + arrayQueue.isEmpty());
    }
}
