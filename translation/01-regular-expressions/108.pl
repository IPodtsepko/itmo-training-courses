while (<>) {
    print if /\([^()]*\w+[^()]*\)/
}
