while (<>) {
    s/\([^)]*\)/\(\)/g;
    print;
}
