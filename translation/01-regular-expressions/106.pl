while (<>) {
    print if /(^|\W)[0-9]+($|\W)/;
}
