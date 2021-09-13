#include <iostream>
#include <fstream>
#include <string>
#include <cassert>

class UniqUtility
{
    bool count_mode = false;
    bool repeated_mode = false;
    bool unique_mode = false;

    int count = 0;
    std::string last_line;

public:
    UniqUtility() = default;

    UniqUtility(bool count, bool repeated, bool unique)
        : count_mode(count)
        , repeated_mode(repeated)
        , unique_mode(unique)
    {
    }

    void print_last_line() const
    {
        if (!count || (unique_mode && count > 1) || (repeated_mode && count == 1)) {
            return;
        }
        if (count_mode) {
            std::cout << "      " << count << ' ';
        }
        std::cout << last_line << "\n";
    }

    void process_line(const std::string &current_line)
    {
        if (!count || current_line != last_line) {
            print_last_line();
            last_line = current_line;
            count = 1;
        } else {
            count++;
        }
    }
};

int main(int argc, char **argv)
{
    bool count_mode = false;
    bool repeated_mode = false;
    bool unique_mode = false;

    std::string *input_file = nullptr;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-c" || arg == "--count_mode") {
            count_mode = true;
        } else if (arg == "-d" || arg == "--repeated_mode") {
            repeated_mode = true;
        } else if (arg == "-u" || arg == "--unique_mode") {
            unique_mode = true;
        } else if (arg == "-cd") {
            count_mode = repeated_mode = true;
        } else if (arg == "-cu") {
            count_mode = unique_mode = true;
        } else if (arg == "-cdu") {
            count_mode = repeated_mode = unique_mode = true;
        } else {
            assert(i == argc - 1);
            input_file = new std::string(arg);
        }
    }

    std::istream *input;
    if (input_file) {
        input = new std::fstream(*input_file);
    } else {
        input = &std::cin;
    }

    UniqUtility uniq(count_mode, repeated_mode, unique_mode);

    std::string line;
    while (std::getline(*input, line)) {
        uniq.process_line(line);
    }
    uniq.print_last_line();

    return 0;
}
