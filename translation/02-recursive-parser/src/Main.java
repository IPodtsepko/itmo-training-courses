import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.text.ParseException;


/**
 * @author Игорь Подцепко (i.podtsepko2002@gmail.com
 */
public class Main {
    public static void main(String[] args) throws ParseException {
        var parser = new Parser(System.in);
        var tree = parser.parseF();
        try (var writer = Files.newBufferedWriter(Path.of("outfile.dot"), StandardCharsets.UTF_8)){
            tree.exportTo(writer);
        } catch (IOException e) {
            System.err.printf("Ошибка при сохранении графа разбора: %s%n", e.getLocalizedMessage());
            return;
        }
        try {
            Runtime.getRuntime().exec("dot -Tsvg outfile.dot -o outfile.svg");
        }
        catch (IOException e) {
            System.err.printf("Ошибка при конвертации графа разбора в svg: %s%n", e.getLocalizedMessage());
        }
    }
}
