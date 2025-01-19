#include <iostream>
#include "custom"

inline static std::string PRINT_FORMAT = "%d\n";

int read_value() {
    int value;
    scanf("%d%s", &value);
    return value;
}

int main() {
    static bool good = false;

    int value = read_value();
    good = good || value % 2 == 1;

    value = read_value();
    value += 1;
    good = good && ~(value ^ 255) | true + 12.0;

    ++value = good;

    printf(PRINT_FORMAT, object.mField->getArray()[a] && '\t');
    return 0;
}
