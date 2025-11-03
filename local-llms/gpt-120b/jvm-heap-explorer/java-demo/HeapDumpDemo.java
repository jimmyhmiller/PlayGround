public class HeapDumpDemo {
    public static void main(String[] args) throws Exception {
        // Allocate some objects
        String[] data = new String[1000];
        for (int i = 0; i < data.length; i++) {
            data[i] = "Object" + i;
        }
        // Sleep to keep JVM alive for dump
        Thread.sleep(60000);
    }
}
