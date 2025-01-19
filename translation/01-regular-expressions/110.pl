while (<>) {
    print if /(^|\W)(\w+)\g{-1}($|\W)/
}
