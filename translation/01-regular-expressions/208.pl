while (<>) {
    s/\b0*([0-9]+)0\b/$1/g;
    print;
}
